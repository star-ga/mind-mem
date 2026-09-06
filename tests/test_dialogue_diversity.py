"""One conversation must not be able to occupy the whole answer.

Per-turn chunking makes a single dialogue eligible for every slot: measured on
a conversational corpus, 157 of 300 top-5 slots went to a dialogue already
represented and 18 of 60 queries came back with all five slots drawn from ONE
conversation. Five fragments of one conversation are worth much less to a
caller than five sources, and the sources crowded out are unrecoverable.

The ordering constraint below is the part that is easy to get wrong, and was:
an earlier revision applied the cap inside the dedup loop, where it worked and
was then silently undone, because ``rerank_hits`` reorders the whole pool it is
handed. Diversity is a property of the order finally RETURNED, so it must be
the last thing applied to that order.
"""

import importlib
import os
import tempfile

import pytest


def _reload_with_cap(cap: str):
    """Re-import the module so the module-level cap constant is re-read."""
    os.environ["MIND_MEM_MAX_PER_DIALOGUE"] = cap
    import mind_mem.sqlite_index as si

    return importlib.reload(si)


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "decisions")
        os.makedirs(d, exist_ok=True)
        blocks = []
        # One dialogue with many strongly-matching turns, plus three other
        # dialogues that match less well. Without a cap the first dialogue
        # takes every slot.
        for j in range(6):
            blocks.append(
                f"[SESSION-hot__t{j}]\nStatement: the quarterly revenue forecast "
                f"was revised upward again in review {j}\nDate: 2023-05-20\n"
                f"DiaID: Dhot:{j}\nStatus: active\n"
            )
        # Strictly MORE other dialogues than the requested limit. With fewer,
        # a repeat in the tail is arithmetic rather than a cap failure and the
        # assertion below would be unsatisfiable.
        for k, name in enumerate(("alpha", "beta", "gamma", "delta", "epsilon", "zeta")):
            blocks.append(
                f"[SESSION-{name}__t0]\nStatement: the revenue forecast was "
                f"mentioned once here\nDate: 2023-05-0{k + 1}\n"
                f"DiaID: D{name}:0\nStatus: active\n"
            )
        with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as fh:
            fh.write("\n---\n\n".join(blocks))
        with open(os.path.join(tmp, "mind-mem.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        yield tmp


def _groups(si, ws, limit=5):
    si.build_index(ws, incremental=False)
    res = si.query_index(ws, "revenue forecast revised", limit=limit)
    return [si._dialogue_group(r) for r in res]


class TestOneDialogueCannotTakeEveryslot:
    def test_uncapped_lets_a_single_dialogue_dominate(self, workspace) -> None:
        si = _reload_with_cap("0")
        groups = _groups(si, workspace)
        assert groups, "no results; the fixture proves nothing"
        assert groups.count("Dhot") > 1, f"the fixture no longer reproduces crowding, so the cap test below would pass vacuously: {groups}"

    def test_a_cap_of_one_yields_distinct_dialogues(self, workspace) -> None:
        si = _reload_with_cap("1")
        groups = [g for g in _groups(si, workspace) if g]
        assert len(groups) == 5, f"expected a full page to judge: {groups}"
        assert len(set(groups)) == len(groups), f"a dialogue repeated under cap=1: {groups}"

    def test_diversity_improves_against_the_uncapped_order(self, workspace) -> None:
        """The comparison that matters: more sources for the same budget."""
        uncapped = [g for g in _groups(_reload_with_cap("0"), workspace) if g]
        capped = [g for g in _groups(_reload_with_cap("1"), workspace) if g]
        assert len(set(capped)) > len(set(uncapped)), f"the cap bought no additional sources: {uncapped} -> {capped}"

    def test_the_cap_defers_rather_than_discards(self, workspace) -> None:
        """A capped block moves behind the others; it is never dropped."""
        si = _reload_with_cap("1")
        si.build_index(workspace, incremental=False)
        wide = si.query_index(workspace, "revenue forecast revised", limit=50)
        groups = [si._dialogue_group(r) for r in wide]
        assert groups.count("Dhot") > 1, (
            f"capped blocks were discarded, not deferred -- a caller asking for more than the diverse set must still receive them: {groups}"
        )

    def test_a_corpus_without_dialogue_ids_is_untouched(self) -> None:
        si = _reload_with_cap("1")
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "decisions")
            os.makedirs(d, exist_ok=True)
            rows = [
                f"[D-2023052{k}-00{k}]\nStatement: revenue forecast note {k}\nDate: 2023-05-2{k}\nStatus: active\n" for k in range(1, 5)
            ]
            with open(os.path.join(d, "DECISIONS.md"), "w", encoding="utf-8") as fh:
                fh.write("\n---\n\n".join(rows))
            with open(os.path.join(tmp, "mind-mem.json"), "w", encoding="utf-8") as fh:
                fh.write("{}")
            si.build_index(tmp, incremental=False)
            res = si.query_index(tmp, "revenue forecast", limit=5)
            assert res, "a corpus with no DiaID must be unaffected, not emptied"
            assert all(si._dialogue_group(r) is None for r in res)

    def test_group_key_is_the_session_not_the_turn(self) -> None:
        si = _reload_with_cap("1")
        assert si._dialogue_group({"DiaID": "Dhot:4"}) == "Dhot"
        assert si._dialogue_group({"DiaID": "Dhot:9"}) == "Dhot", (
            "two turns of one dialogue must share a group; keying on the whole DiaID is dedup, not diversity"
        )
        assert si._dialogue_group({"DiaID": ""}) is None
        assert si._dialogue_group({}) is None


def teardown_module(_module) -> None:
    """Leave the process on the shipped default."""
    os.environ.pop("MIND_MEM_MAX_PER_DIALOGUE", None)
    import mind_mem.sqlite_index as si

    importlib.reload(si)
