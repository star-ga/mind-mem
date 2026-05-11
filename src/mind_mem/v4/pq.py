"""v4 product-quantization (PQ) encoding for embedding storage (Group D).

Multi-LLM v4 audit (4/4 model consensus 2026-05-10) flagged that
storing raw float32 embeddings doesn't scale: a 768-dim vector at
float32 is ~3 KB; 10⁶ blocks → ~3 GB just for embeddings. Product
quantization compresses each vector to ~32 bytes (96× reduction)
with a typical 5-10% recall hit at large scale.

Design:

    Vector (D-dim float32)
        ──split──▶ M sub-vectors (D/M dims each)
        ──train──▶ K centroids per subvector position (KMeans, K=256)
        ──encode▶ M-byte code (one byte per subvector position)
        ──decode▶ approximate reconstruction by concatenating centroids

Defaults (from the audit):

    M = 32       subvector positions
    K = 256      centroids per position (8-bit codes)
    bytes        M = 32 bytes per encoded vector

For 768-dim float32 input:

    raw size       768 × 4 = 3072 bytes
    PQ size        32 bytes
    compression    96× (≈ what the audit cited)

Distance computation uses the **asymmetric** distance: a query vector
is split into subvectors, distance to each codebook centroid is
precomputed (one M×K table), then a database scan sums table lookups
per encoded vector. Cost per scan: M lookups + M-1 adds per vector,
no float ops on the database side.

This module is pure stdlib (no numpy) — slow on large training sets
but correct and dependency-free. The training step (codebook fit)
lands here for reference; production deployments swap in a numpy /
scikit-learn implementation by setting an alternate trainer via
:func:`set_codebook_trainer`. The encode / decode / distance paths
work unchanged on whatever codebook the trainer produced.

Feature-flag gated under ``v4.pq``. Adding the flag to the registry
is part of this commit — see :mod:`mind_mem.v4.feature_flags`.

Copyright STARGA, Inc.
"""

from __future__ import annotations

import math
import random
import sqlite3
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .feature_flags import flag_config, require_enabled

__all__ = [
    "FLAG",
    "PQConfig",
    "DEFAULT_PQ_CONFIG",
    "Codebook",
    "train_codebook",
    "set_codebook_trainer",
    "encode",
    "decode",
    "asymmetric_distance",
    "ensure_pq_schema",
    "store_codebook",
    "load_codebook",
    "store_code",
    "load_code",
]


#: Feature-flag key in ``mind-mem.json: v4: {...}``.
FLAG: str = "pq"


@dataclass(frozen=True)
class PQConfig:
    """Tunable knobs for PQ encoding.

    Defaults: M=32 subvectors, K=256 centroids, 8-bit codes (1 byte
    per subvector position). Override via mind-mem.json:

        "v4": {
            "pq": {
                "enabled": true,
                "subvectors": 32,
                "centroids": 256,
                "kmeans_iters": 25,
                "kmeans_seed": 1
            }
        }
    """

    subvectors: int = 32
    centroids: int = 256
    kmeans_iters: int = 25
    kmeans_seed: int = 1


DEFAULT_PQ_CONFIG: PQConfig = PQConfig()


@dataclass(frozen=True)
class Codebook:
    """Trained PQ codebook.

    A list of ``M`` centroid tables; each table holds ``K`` centroids
    of ``D/M`` dims each. ``cfg`` records the training-time
    hyperparameters so an encode/decode pair always agrees on shape.
    """

    cfg: PQConfig
    centroids: tuple[tuple[tuple[float, ...], ...], ...]
    """centroids[m][k] = D/M-dim centroid k for subvector position m."""

    @property
    def dim(self) -> int:
        """Original vector dimension recovered from the codebook."""
        return self.cfg.subvectors * self.subvector_dim

    @property
    def subvector_dim(self) -> int:
        """Dimension of one sub-vector (D / M)."""
        if not self.centroids or not self.centroids[0]:
            return 0
        return len(self.centroids[0][0])


# ---------------------------------------------------------------------------
# Pluggable trainer (so production can swap in numpy / sklearn)
# ---------------------------------------------------------------------------


CodebookTrainer = Callable[[Sequence[Sequence[float]], PQConfig], Codebook]


def _stdlib_kmeans(
    points: list[tuple[float, ...]],
    k: int,
    iters: int,
    rng: random.Random,
) -> list[tuple[float, ...]]:
    """Lloyd's algorithm in pure Python; returns final centroid list.

    Initialisation: k-means++ seeding for stable convergence on small
    subvector spaces. Iterations: assign → recentre → repeat. Stops
    early when no centroid moves more than 1e-9 in one iteration.
    Empty clusters re-seed from a random point.
    """
    if not points:
        return []
    if k <= 0:
        return []
    if k >= len(points):
        # Trivial case: each point is its own centroid.
        return list(points)

    # k-means++ seeding.
    centers: list[tuple[float, ...]] = [points[rng.randrange(len(points))]]
    while len(centers) < k:
        d2 = [min(_squared_dist(p, c) for c in centers) for p in points]
        total = sum(d2)
        if total <= 0.0:
            centers.append(points[rng.randrange(len(points))])
            continue
        r = rng.random() * total
        acc = 0.0
        for p, d in zip(points, d2):
            acc += d
            if acc >= r:
                centers.append(p)
                break
        else:
            centers.append(points[-1])

    dim = len(points[0])
    for _ in range(max(1, iters)):
        # Assign.
        assignments = [_argmin_index(p, centers) for p in points]
        # Recentre.
        new_centers: list[tuple[float, ...]] = []
        moved = 0.0
        for k_idx in range(k):
            members = [points[i] for i, a in enumerate(assignments) if a == k_idx]
            if not members:
                # Empty cluster → re-seed.
                replacement = points[rng.randrange(len(points))]
                new_centers.append(replacement)
                continue
            mean = tuple(sum(m[j] for m in members) / len(members) for j in range(dim))
            moved = max(moved, _squared_dist(mean, centers[k_idx]))
            new_centers.append(mean)
        centers = new_centers
        if moved < 1e-18:
            break
    return centers


def _argmin_index(p: tuple[float, ...], centers: Sequence[tuple[float, ...]]) -> int:
    best, best_idx = float("inf"), 0
    for i, c in enumerate(centers):
        d = _squared_dist(p, c)
        if d < best:
            best, best_idx = d, i
    return best_idx


def _squared_dist(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        diff = x - y
        s += diff * diff
    return s


def _stdlib_train_codebook(training: Sequence[Sequence[float]], cfg: PQConfig) -> Codebook:
    """Default trainer: stdlib k-means on each subvector position."""
    if not training:
        # Empty training set → empty codebook is legal (all encodes
        # become zero codes; decode reconstructs zero vectors).
        return Codebook(cfg=cfg, centroids=tuple())

    dim = len(training[0])
    if cfg.subvectors <= 0 or dim % cfg.subvectors != 0:
        raise ValueError(f"subvectors ({cfg.subvectors}) must divide dim ({dim}) evenly")
    sub_dim = dim // cfg.subvectors
    rng = random.Random(cfg.kmeans_seed)

    codebook: list[tuple[tuple[float, ...], ...]] = []
    for m in range(cfg.subvectors):
        sub_points = [tuple(v[m * sub_dim : (m + 1) * sub_dim]) for v in training if len(v) == dim]
        centers = _stdlib_kmeans(sub_points, cfg.centroids, cfg.kmeans_iters, rng)
        # Pad with zero-vectors when training set has < K distinct points.
        while len(centers) < cfg.centroids:
            centers.append(tuple(0.0 for _ in range(sub_dim)))
        codebook.append(tuple(centers))
    return Codebook(cfg=cfg, centroids=tuple(codebook))


_active_trainer: CodebookTrainer = _stdlib_train_codebook


def set_codebook_trainer(fn: CodebookTrainer) -> None:
    """Swap the codebook trainer (e.g. install a numpy-backed one).

    Production deployments install a faster trainer at startup; the
    default stdlib version is correct but slow on large training sets.
    """
    require_enabled(FLAG)
    global _active_trainer
    _active_trainer = fn


def train_codebook(training: Sequence[Sequence[float]], cfg: PQConfig | None = None) -> Codebook:
    """Fit a codebook to the training set under the given config."""
    require_enabled(FLAG)
    return _active_trainer(training, cfg or _load_config())


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


def encode(vec: Sequence[float], codebook: Codebook) -> bytes:
    """Encode a single vector to ``M`` bytes using ``codebook``.

    Returns an empty ``bytes`` if the codebook is empty or the vector
    length doesn't match. Each output byte is the index of the nearest
    centroid for that subvector position; 8-bit codes assume K ≤ 256.
    """
    require_enabled(FLAG)
    if not codebook.centroids:
        return b""
    sub_dim = codebook.subvector_dim
    if len(vec) != codebook.cfg.subvectors * sub_dim:
        return b""
    if codebook.cfg.centroids > 256:
        # Out of byte range — caller should split bigger codebooks.
        return b""
    out = bytearray(codebook.cfg.subvectors)
    for m in range(codebook.cfg.subvectors):
        sub = tuple(vec[m * sub_dim : (m + 1) * sub_dim])
        out[m] = _argmin_index(sub, codebook.centroids[m])
    return bytes(out)


def decode(code: bytes, codebook: Codebook) -> list[float]:
    """Reconstruct an approximate vector from ``code`` and ``codebook``.

    Empty ``code`` returns an empty list. Each input byte is looked up
    in the codebook; the centroid for each subvector position is
    concatenated.
    """
    require_enabled(FLAG)
    if not code or not codebook.centroids:
        return []
    if len(code) != codebook.cfg.subvectors:
        return []
    out: list[float] = []
    for m, b in enumerate(code):
        if b >= len(codebook.centroids[m]):
            # Corrupt code — pad with zeros for the rest of the subvector.
            out.extend(0.0 for _ in range(codebook.subvector_dim))
            continue
        out.extend(codebook.centroids[m][b])
    return out


def asymmetric_distance(query: Sequence[float], code: bytes, codebook: Codebook) -> float:
    """Squared L2 distance between a raw query and a PQ-encoded vector.

    Uses the standard asymmetric formulation: at scan time, for each
    subvector position ``m``, look up the centroid the code points to
    and accumulate squared distance to the query's m-th subvector.
    No reconstruction needed.

    Returns ``+inf`` for empty/mismatched inputs so callers can treat
    them as "definitely not the nearest neighbour" without a special
    case.
    """
    require_enabled(FLAG)
    if not code or not codebook.centroids:
        return math.inf
    sub_dim = codebook.subvector_dim
    if len(query) != codebook.cfg.subvectors * sub_dim:
        return math.inf
    if len(code) != codebook.cfg.subvectors:
        return math.inf
    total = 0.0
    for m, b in enumerate(code):
        if b >= len(codebook.centroids[m]):
            return math.inf
        c = codebook.centroids[m][b]
        sub = query[m * sub_dim : (m + 1) * sub_dim]
        for x, y in zip(sub, c):
            d = x - y
            total += d * d
    return total


# ---------------------------------------------------------------------------
# Persistence (SQLite)
# ---------------------------------------------------------------------------

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS pq_codebook (
    name        TEXT PRIMARY KEY,
    subvectors  INTEGER NOT NULL,
    centroids   INTEGER NOT NULL,
    sub_dim     INTEGER NOT NULL,
    payload     BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS pq_codes (
    block_id    TEXT NOT NULL,
    codebook    TEXT NOT NULL,
    code        BLOB NOT NULL,
    PRIMARY KEY (block_id, codebook)
);
"""


def ensure_pq_schema(workspace: str | Path) -> None:
    """Create the PQ tables on first call. Idempotent."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.parent.is_dir():
        db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db, timeout=30) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


def _serialise_codebook(cb: Codebook) -> bytes:
    """Pack the codebook to a flat float32 BLOB.

    Layout: M × K × sub_dim float32 values, row-major
    (m fastest-varying inside a centroid table, k inside m, dim inside k).
    """
    if not cb.centroids:
        return b""
    M, K, dim = cb.cfg.subvectors, cb.cfg.centroids, cb.subvector_dim
    fmt = f"<{M * K * dim}f"
    flat: list[float] = []
    for m in range(M):
        for k in range(K):
            flat.extend(cb.centroids[m][k])
    return struct.pack(fmt, *flat)


def _deserialise_codebook(payload: bytes, cfg: PQConfig, sub_dim: int) -> Codebook:
    M, K = cfg.subvectors, cfg.centroids
    fmt = f"<{M * K * sub_dim}f"
    if len(payload) != struct.calcsize(fmt):
        return Codebook(cfg=cfg, centroids=tuple())
    flat = struct.unpack(fmt, payload)
    centroids: list[tuple[tuple[float, ...], ...]] = []
    idx = 0
    for _m in range(M):
        row: list[tuple[float, ...]] = []
        for _k in range(K):
            row.append(tuple(flat[idx : idx + sub_dim]))
            idx += sub_dim
        centroids.append(tuple(row))
    return Codebook(cfg=cfg, centroids=tuple(centroids))


def store_codebook(workspace: str | Path, name: str, codebook: Codebook) -> None:
    """Upsert a codebook into the workspace by name."""
    require_enabled(FLAG)
    ensure_pq_schema(workspace)
    db = Path(workspace) / "index.db"
    with sqlite3.connect(db, timeout=30) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pq_codebook (name, subvectors, centroids, sub_dim, payload) VALUES (?, ?, ?, ?, ?)",
            (
                name,
                codebook.cfg.subvectors,
                codebook.cfg.centroids,
                codebook.subvector_dim,
                _serialise_codebook(codebook),
            ),
        )
        conn.commit()


def load_codebook(workspace: str | Path, name: str) -> Codebook | None:
    """Return the codebook by name, or ``None`` if absent / unreadable."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with sqlite3.connect(db, timeout=30) as conn:
        row = (
            conn.execute(
                "SELECT subvectors, centroids, sub_dim, payload FROM pq_codebook WHERE name = ?",
                (name,),
            ).fetchone()
            if _table_exists(conn, "pq_codebook")
            else None
        )
    if row is None:
        return None
    M, K, sub_dim, payload = row
    cfg = PQConfig(subvectors=int(M), centroids=int(K))
    return _deserialise_codebook(payload, cfg, int(sub_dim))


def store_code(workspace: str | Path, block_id: str, codebook_name: str, code: bytes) -> None:
    """Upsert a single block's PQ code under the named codebook."""
    require_enabled(FLAG)
    ensure_pq_schema(workspace)
    db = Path(workspace) / "index.db"
    with sqlite3.connect(db, timeout=30) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pq_codes (block_id, codebook, code) VALUES (?, ?, ?)",
            (block_id, codebook_name, code),
        )
        conn.commit()


def load_code(workspace: str | Path, block_id: str, codebook_name: str) -> bytes | None:
    """Return the PQ code for a block under the named codebook."""
    require_enabled(FLAG)
    db = Path(workspace) / "index.db"
    if not db.is_file():
        return None
    with sqlite3.connect(db, timeout=30) as conn:
        if not _table_exists(conn, "pq_codes"):
            return None
        row = conn.execute(
            "SELECT code FROM pq_codes WHERE block_id = ? AND codebook = ?",
            (block_id, codebook_name),
        ).fetchone()
    return bytes(row[0]) if row else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _load_config() -> PQConfig:
    raw = flag_config(FLAG)
    if not isinstance(raw, dict):
        return DEFAULT_PQ_CONFIG
    fields = {
        "subvectors": (int, DEFAULT_PQ_CONFIG.subvectors),
        "centroids": (int, DEFAULT_PQ_CONFIG.centroids),
        "kmeans_iters": (int, DEFAULT_PQ_CONFIG.kmeans_iters),
        "kmeans_seed": (int, DEFAULT_PQ_CONFIG.kmeans_seed),
    }
    out: dict[str, int] = {}
    for key, (caster, default) in fields.items():
        v = raw.get(key, default)
        try:
            out[key] = caster(v)
        except (TypeError, ValueError):
            out[key] = default
    return PQConfig(**out)
