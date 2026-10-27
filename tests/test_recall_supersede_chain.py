"""A superseded memory never outranks, and is never returned instead of, its superseder.

Every scenario is a workspace the blind verifier built (s01..s29, t17, t18,
t30 and the custom-namespace leak probe), served on every recall leg: the
in-memory scan (library and MCP), the indexed sqlite backend through
``recall``, ``query_index``, ``HybridBackend.search`` and the MCP tool's
indexed backends.

The contract, per leg:

* a returned block that a caller-readable, admitted, live block supersedes
  has the live head of its chain directly above it, even when the head shares
  no query term with the question;
* chain members keep chain order (C, B, A for C <- B <- A);
* ids match whole known ids only (``X-r2`` does not retire ``X``);
* quarantined / pending superseders and empty markers retire nothing;
* a superseder the caller cannot read neither retires nor reveals anything;
* cycles terminate and retire nothing on their own.

Each negative assertion is paired with a positive control: the block that
must NOT be retired or revealed is shown to exist (and to be served to a
caller that may read it, or on a corpus without the link).
"""

from __future__ import annotations

import pytest
from _supersede_scenarios import (
    ACL_SHARED_AND_ROOT,
    ACL_SHARED_ONLY,
    ALL_LEGS,
    CUSTOM_NS,
    INDEX_LEGS,
    STRONG,
    TGT,
    VERYWEAK,
    WEAK,
    Scenario,
    Workspaces,
    blk,
)

SUP = "D-20261008-001"
A, B, C = "D-20261001-001", "D-20261002-001", "D-20261003-001"


@pytest.fixture(scope="module")
def spaces(tmp_path_factory: pytest.TempPathFactory) -> Workspaces:
    return Workspaces(str(tmp_path_factory.mktemp("supersede")))


def _above(lst: list[str], upper: str, lower: str) -> None:
    assert upper in lst, f"{upper} missing from {lst}"
    if lower in lst:
        assert lst.index(upper) < lst.index(lower), lst


def _directly_above(lst: list[str], upper: str, lower: str) -> None:
    assert upper in lst and lower in lst, lst
    assert lst.index(upper) + 1 == lst.index(lower), lst


# ---------------------------------------------------------------------------
# One superseder, one target (s01..s05, s22..s26)
# ---------------------------------------------------------------------------

LINKED = {
    "s01_sup_sorts_first": (
        "D-20261001-001",
        {"decisions/DECISIONS.md": blk(TGT, STRONG) + blk("D-20261001-001", WEAK, extra=f"Supersedes: {TGT}\n")},
    ),
    "s02_sup_sorts_last": (
        "D-20261009-009",
        {"decisions/DECISIONS.md": blk("D-20261009-009", WEAK, extra=f"Supersedes: {TGT}\n") + blk(TGT, STRONG)},
    ),
    "s03_low_lexical": (SUP, {"decisions/DECISIONS.md": blk(TGT, STRONG) + blk(SUP, VERYWEAK, extra=f"Supersedes: {TGT}\n")}),
    "s04_list_form": (
        SUP,
        {"decisions/DECISIONS.md": blk(SUP, WEAK, extra=f"Supersedes:\n- D-20250101-001\n- {TGT}\n") + blk(TGT, STRONG)},
    ),
    "s05_several_ids": (
        SUP,
        {"decisions/DECISIONS.md": blk(SUP, WEAK, extra=f"Supersedes: D-20250101-001; {TGT} D-20250101-002\n") + blk(TGT, STRONG)},
    ),
    "s22_nested_constraint_sig": (
        SUP,
        {
            "decisions/DECISIONS.md": blk(TGT, STRONG)
            + blk(SUP, WEAK, extra=f"ConstraintSignatures:\n- id: CS-001\n  domain: schedule\n  supersedes: {TGT}\n")
        },
    ),
    "s25_sup_no_query_terms": (
        SUP,
        {"decisions/DECISIONS.md": blk(TGT, STRONG) + blk(SUP, "Moved to Wednesday.", extra=f"Supersedes: {TGT}\n")},
    ),
}


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize("name", sorted(LINKED))
def test_superseder_is_directly_above_its_target(spaces: Workspaces, name: str, leg: str) -> None:
    sup, files = LINKED[name]
    got = spaces.serve(Scenario(name, files), leg)
    _directly_above(got, sup, TGT)


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize("limit", [1, 2])
def test_superseded_block_is_never_returned_instead_of_its_superseder(spaces: Workspaces, leg: str, limit: int) -> None:
    sup, files = LINKED["s25_sup_no_query_terms"]
    got = spaces.serve(Scenario("s25_sup_no_query_terms", files), leg, limit=limit)
    assert got[0] == sup, got
    # Positive control: the same corpus without the link serves the target first.
    control = spaces.serve(Scenario("s25_sup_no_query_terms", files), leg, limit=10)
    _directly_above(control, sup, TGT)


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_lowercase_top_level_field_is_not_a_field(spaces: Workspaces, leg: str) -> None:
    """s23: the block grammar reads ``Capitalised:`` fields only.

    ``block_parser`` drops a lowercase top-level line for EVERY field (a
    lowercase ``statement:`` is not a statement either), so a lowercase
    ``supersedes:`` is no link on any leg. Nested lowercase keys
    (``ConstraintSignatures[].supersedes``) are parsed and do link (s22).
    """
    files = {"decisions/DECISIONS.md": blk(TGT, STRONG) + blk(SUP, WEAK, extra=f"supersedes: {TGT}\n")}
    got = spaces.serve(Scenario("s23_lowercase_field", files), leg)
    assert got[:2] == [TGT, SUP], got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_cross_file_target(spaces: Workspaces, leg: str) -> None:
    files = {
        "tasks/TASKS.md": blk("T-20261006-001", STRONG, kind="Task"),
        "decisions/DECISIONS.md": blk(SUP, WEAK, extra="Supersedes: T-20261006-001\n"),
    }
    got = spaces.serve(Scenario("s24_cross_file", files), leg)
    _directly_above(got, SUP, "T-20261006-001")


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_non_standard_ids_link_as_whole_ids(spaces: Workspaces, leg: str) -> None:
    files = {"decisions/DECISIONS.md": blk("DEC-0001", STRONG) + blk("DEC-0002", WEAK, extra="Supersedes: DEC-0001\n")}
    got = spaces.serve(Scenario("s26_nonstd_ids", files), leg)
    _directly_above(got, "DEC-0002", "DEC-0001")


# ---------------------------------------------------------------------------
# Several targets / crowded windows (s06..s08)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_one_superseder_of_several_targets_is_above_all_of_them(spaces: Workspaces, leg: str) -> None:
    targets = [f"D-2026100{i}-00{i}" for i in range(2, 6)]
    body = "".join(blk(t, STRONG) for t in targets) + blk(SUP, WEAK, extra="Supersedes: " + ", ".join(targets) + "\n")
    got = spaces.serve(Scenario("s06_several_targets", {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == SUP, got
    for target in targets:
        assert target in got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_window_full_of_link_superseded_rows_serves_the_superseder_first(spaces: Workspaces, leg: str) -> None:
    targets = [f"D-20260901-{i:03d}" for i in range(1, 41)]
    body = "".join(blk(t, STRONG) for t in targets) + blk(SUP, WEAK, extra="Supersedes:\n" + "".join(f"- {t}\n" for t in targets))
    got = spaces.serve(Scenario("s07_window_40_link", {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == SUP, got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_window_mixed_dead_status_and_link(spaces: Workspaces, leg: str) -> None:
    linked = [f"D-20260902-{i:03d}" for i in range(1, 21)]
    dead = [f"D-20260903-{i:03d}" for i in range(1, 21)]
    body = "".join(blk(t, STRONG) for t in linked) + "".join(blk(t, STRONG, status="superseded") for t in dead)
    body += blk(SUP, WEAK, extra="Supersedes: " + " ".join(linked) + "\n")
    got = spaces.serve(Scenario("s08_window_mixed", {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == SUP, got


# ---------------------------------------------------------------------------
# Chains and cycles (s09..s12)
# ---------------------------------------------------------------------------

CHAINS = {
    "s09_chain_strongA": blk(A, STRONG)
    + blk(B, STRONG + " moved", extra=f"Supersedes: {A}\n")
    + blk(C, VERYWEAK + " weekly", extra=f"Supersedes: {B}\n"),
    "s10_chain_A_above_B": blk(A, STRONG)
    + blk(B, WEAK, extra=f"Supersedes: {A}\n")
    + blk(C, VERYWEAK + " weekly", extra=f"Supersedes: {B}\n"),
}


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize("name", sorted(CHAINS))
def test_chain_keeps_chain_order(spaces: Workspaces, name: str, leg: str) -> None:
    got = spaces.serve(Scenario(name, {"decisions/DECISIONS.md": CHAINS[name]}), leg)
    assert got[:3] == [C, B, A], got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_chain_head_is_pulled_in_when_only_the_tail_matches(spaces: Workspaces, leg: str) -> None:
    body = (
        blk(A, STRONG)
        + blk(B, "Moved to Wednesday.", extra=f"Supersedes: {A}\n")
        + blk(C, "Moved to Thursday.", extra=f"Supersedes: {B}\n")
    )
    got = spaces.serve(Scenario("chain_tail_only", {"decisions/DECISIONS.md": body}), leg)
    # The head C shares no query term; it is pulled in directly above A.
    # B matches nothing either and is not the head, so it is not pulled.
    assert got[:2] == [C, A], got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_two_cycle_terminates_and_retires_nothing(spaces: Workspaces, leg: str) -> None:
    body = blk(A, STRONG, extra=f"Supersedes: {B}\n") + blk(B, WEAK, extra=f"Supersedes: {A}\n")
    got = spaces.serve(Scenario("s11_cycle2", {"decisions/DECISIONS.md": body}), leg)
    assert set(got) == {A, B}, got
    assert got[0] == A, got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_three_cycle_terminates(spaces: Workspaces, leg: str) -> None:
    body = (
        blk(A, STRONG, extra=f"Supersedes: {C}\n")
        + blk(B, WEAK, extra=f"Supersedes: {A}\n")
        + blk(C, WEAK, extra=f"Supersedes: {B}\n")
        + blk("D-20261004-001", "weekly sync notes")
    )
    got = spaces.serve(Scenario("s12_cycle3", {"decisions/DECISIONS.md": body}), leg)
    assert {A, B, C} <= set(got), got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_cycle_with_a_live_exit_resolves_to_the_live_head(spaces: Workspaces, leg: str) -> None:
    body = (
        blk(A, STRONG, extra=f"Supersedes: {B}\n")
        + blk(B, WEAK, extra=f"Supersedes: {A}\n")
        + blk(C, "Moved to Thursday.", extra=f"Supersedes: {A}\n")
    )
    got = spaces.serve(Scenario("cycle_live_exit", {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == C, got


# ---------------------------------------------------------------------------
# What does NOT retire (s13..s15, s27..s29, t30)
# ---------------------------------------------------------------------------

WITHHELD = {
    "s13_quarantined_sup": "quarantined",
    "s14_pending_sup": "pending",
    "s28_quarantined_caps": "Quarantined",
}


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize("name", sorted(WITHHELD))
def test_withheld_superseder_retires_nothing(spaces: Workspaces, name: str, leg: str) -> None:
    body = blk(SUP, WEAK, status=WITHHELD[name], extra=f"Supersedes: {TGT}\n") + blk(TGT, STRONG) + blk("D-20261009-001", "weekly archive")
    got = spaces.serve(Scenario(name, {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == TGT, got
    assert SUP not in got, got
    # Positive control: the same superseder, active, does retire the target.
    active = blk(SUP, WEAK, extra=f"Supersedes: {TGT}\n") + blk(TGT, STRONG) + blk("D-20261009-001", "weekly archive")
    control = spaces.serve(Scenario(name + "_active", {"decisions/DECISIONS.md": active}), leg)
    _directly_above(control, SUP, TGT)


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_empty_markers_are_not_links(spaces: Workspaces, leg: str) -> None:
    body = blk(TGT, STRONG)
    for i, marker in enumerate(["none", "null", "n/a", "-", "", "None", "N/A", "NULL"], start=1):
        body += blk(f"D-20261008-{i:03d}", WEAK, extra=f"Supersedes: {marker}\n")
    got = spaces.serve(Scenario("s15_empty_markers", {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == TGT, got
    assert "D-20261008-001" in got  # control: the marker blocks are served


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_suffixed_id_does_not_retire_its_prefix(spaces: Workspaces, leg: str) -> None:
    body = blk(TGT, STRONG) + blk(SUP, WEAK, extra=f"Supersedes: {TGT}-r2\n")
    got = spaces.serve(Scenario("s27_suffix_id_false_link", {"decisions/DECISIONS.md": body}), leg)
    assert got[0] == TGT, got
    assert SUP in got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_revision_suffix_retires_only_the_named_revision(spaces: Workspaces, leg: str) -> None:
    rev = TGT + "-r2"
    body = blk(TGT, STRONG) + blk(rev, STRONG + " revised") + blk(SUP, WEAK, extra=f"Supersedes: {rev}\n")
    got = spaces.serve(Scenario("t30_revision_suffix", {"decisions/DECISIONS.md": body}), leg)
    # The unsuffixed block scores above the revision. Were it retired by a
    # prefix match, SUP would be pulled above IT and could not also sit
    # directly above the revision.
    assert got.index(TGT) < got.index(rev), got
    _directly_above(got, SUP, rev)


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_no_link_control_keeps_score_order(spaces: Workspaces, leg: str) -> None:
    got = spaces.serve(Scenario("s29_nolink_control", {"decisions/DECISIONS.md": blk(TGT, STRONG) + blk(SUP, WEAK)}), leg)
    assert got[:2] == [TGT, SUP], got


# ---------------------------------------------------------------------------
# Namespaces and the caller (s16..s21, t17, t18, leak)
# ---------------------------------------------------------------------------


def _ns_files(rel: str) -> dict[str, str]:
    return {"decisions/DECISIONS.md": blk(TGT, STRONG), rel: blk(SUP, WEAK, extra=f"Supersedes: {TGT}\n")}


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize(
    "name,rel",
    [("s16_shared_unbound", "shared/decisions/DECISIONS.md"), ("s19_agentfile_unbound", "agents/a1/decisions/DECISIONS.md")],
)
def test_unbound_caller_is_not_served_agent_namespaces_and_they_retire_nothing(spaces: Workspaces, name: str, rel: str, leg: str) -> None:
    sc = Scenario(name, _ns_files(rel), acl=ACL_SHARED_ONLY)
    got = spaces.serve(sc, leg)
    assert got[0] == TGT, got
    assert SUP not in got, got
    # Positive control: a bound caller that reads that namespace is served the superseder.
    bound = spaces.serve(sc, leg, agent="a1")
    assert SUP in bound, bound


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize(
    "name,rel",
    [("s17_shared_bound", "shared/decisions/DECISIONS.md"), ("s18_agentfile_bound", "agents/a1/decisions/DECISIONS.md")],
)
def test_bound_caller_without_root_read_is_served_its_namespace(spaces: Workspaces, name: str, rel: str, leg: str) -> None:
    got = spaces.serve(Scenario(name, _ns_files(rel), acl=ACL_SHARED_ONLY, agent="a1"), leg)
    assert got == [SUP], got


@pytest.mark.parametrize("leg", ALL_LEGS)
@pytest.mark.parametrize(
    "name,rel",
    [("t17_shared_bound_rootread", "shared/decisions/DECISIONS.md"), ("t18_agentfile_bound_rootread", "agents/a1/decisions/DECISIONS.md")],
)
def test_bound_caller_gets_namespace_edges(spaces: Workspaces, name: str, rel: str, leg: str) -> None:
    got = spaces.serve(Scenario(name, _ns_files(rel), acl=ACL_SHARED_AND_ROOT, agent="a1"), leg)
    _directly_above(got, SUP, TGT)


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_declared_custom_namespace_superseder(spaces: Workspaces, leg: str) -> None:
    got = spaces.serve(Scenario("s20_custom_ns_declared", _ns_files("custom/decisions/DECISIONS.md"), recall_cfg=CUSTOM_NS), leg)
    _directly_above(got, SUP, TGT)


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_undeclared_custom_directory_is_not_corpus(spaces: Workspaces, leg: str) -> None:
    got = spaces.serve(Scenario("s21_custom_ns_undeclared", _ns_files("custom/decisions/DECISIONS.md")), leg)
    assert got[0] == TGT, got
    assert SUP not in got, got


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_unreadable_superseder_neither_retires_nor_reveals(spaces: Workspaces, leg: str) -> None:
    acl = {"default_policy": "read", "agents": {"a1": {"namespaces": ["agents/a1"], "write": [], "read": ["decisions", "agents/a1"]}}}
    files = {
        "decisions/DECISIONS.md": blk(TGT, STRONG) + blk("D-20261007-001", "weekly call notes"),
        "custom/decisions/DECISIONS.md": blk(SUP, "hidden replacement", extra=f"Supersedes: {TGT}\n"),
    }
    sc = Scenario("leak", files, recall_cfg=CUSTOM_NS, acl=acl, agent="a1")
    got = spaces.serve(sc, leg, limit=5)
    assert got[0] == TGT, got
    assert SUP not in got, got
    # Positive control: an unbound caller reads the declared namespace and is
    # served the replacement directly above the target.
    unbound = spaces.serve(sc, leg, agent="", limit=5)
    _directly_above(unbound, SUP, TGT)


def test_index_legs_cover_every_leg_name() -> None:
    assert set(INDEX_LEGS) < set(ALL_LEGS)


@pytest.mark.parametrize("leg", ALL_LEGS)
def test_unreadable_link_inside_a_chain_reveals_nothing(spaces: Workspaces, leg: str) -> None:
    """TGT <- U <- H with U unreadable: H is readable, but the chain runs through U.

    Following the edge out of U would retire TGT and pull H above it, which
    tells the caller that a block it cannot read replaced TGT.
    """
    acl = {"default_policy": "read", "agents": {"a1": {"namespaces": ["agents/a1"], "write": [], "read": ["decisions", "agents/a1"]}}}
    hidden = "D-20261007-009"
    head = "D-20261009-009"
    files = {
        "decisions/DECISIONS.md": blk(TGT, STRONG)
        + blk("D-20261007-001", "weekly call notes")
        + blk(head, "Moved to Thursday.", extra=f"Supersedes: {hidden}\n"),
        "custom/decisions/DECISIONS.md": blk(hidden, "hidden replacement", extra=f"Supersedes: {TGT}\n"),
    }
    sc = Scenario("leak_chain", files, recall_cfg=CUSTOM_NS, acl=acl, agent="a1")
    got = spaces.serve(sc, leg, limit=5)
    assert got[0] == TGT, got
    assert head not in got and hidden not in got, got
    # Positive control: the unbound caller reads the custom namespace, so the
    # chain resolves and the head is pulled in directly above the target.
    unbound = spaces.serve(sc, leg, agent="", limit=5)
    _directly_above(unbound, head, TGT)
