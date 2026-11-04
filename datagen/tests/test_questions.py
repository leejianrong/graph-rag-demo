"""Questions come from the world model, carry gold paths, and have a unique answer."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from kggen.cli import main
from kggen.questions import VERB, QuestionSpec, generate_questions
from kggen.spec import WorldSpec
from kggen.world import WorldGenerationError, generate


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {"n_entities": 500, "avg_degree": 5, "seed": 2, "alias_rate": 0.5}
    return WorldSpec.model_validate({**base, **overrides})


@pytest.fixture(scope="module")
def world():
    return generate(make(churn_rate=0.1, retraction_rate=0.05, homonym_rate=0.05))


def test_every_question_has_a_gold_path_of_real_final_state_facts(world) -> None:
    current = {e.id: e for e in world.current_edges()}
    for q in generate_questions(world, QuestionSpec(n_questions=60)):
        assert len(q.supporting_facts) == q.hops
        assert all(f in current for f in q.supporting_facts)


def test_the_gold_path_is_a_connected_chain_from_start_to_answer(world) -> None:
    facts = {e.id: e for e in world.edges}
    for q in generate_questions(world, QuestionSpec(n_questions=60)):
        at = q.start
        for fid in q.supporting_facts:
            e = facts[fid]
            assert at in (e.subject, e.object)
            at = e.object if e.subject == at else e.subject
        assert at == q.answer


def test_answers_are_unique_in_the_final_state(world) -> None:
    facts = {e.id: e for e in world.edges}
    current = world.current_edges()
    for q in generate_questions(world, QuestionSpec(n_questions=40)):
        frontier = {q.start}
        for fid in q.supporting_facts:
            e = facts[fid]
            forward = e.subject in frontier
            frontier = {
                (c.object if forward else c.subject)
                for c in current
                if c.predicate == e.predicate and (c.subject if forward else c.object) in frontier
            }
        assert frontier == {q.answer}


def test_hop_distribution_follows_the_spec(world) -> None:
    mix = {1: 1.0, 2: 3.0, 4: 1.0}
    qs = generate_questions(world, QuestionSpec(n_questions=50, hop_mix=mix))
    assert Counter(q.hops for q in qs) == {1: 10, 2: 30, 4: 10}


def test_questions_name_a_uniquely_named_start_and_accept_aliases(world) -> None:
    names = Counter(e.name for e in world.entities)
    by_id = {e.id: e for e in world.entities}
    for q in generate_questions(world, QuestionSpec(n_questions=60)):
        start = by_id[q.start]
        assert names[start.name] == 1 and start.name in q.question
        assert q.accepted_answers[0] == q.answer_name
        assert set(q.accepted_answers[1:]) == set(by_id[q.answer].aliases)
        assert not by_id[q.answer].distractor and not start.distractor


def test_questions_read_from_the_relation_wording(world) -> None:
    facts = {e.id: e for e in world.edges}
    for q in generate_questions(world, QuestionSpec(n_questions=30)):
        assert q.question.startswith("Name the ") and q.question.endswith(".")
        assert all(VERB[facts[f].predicate] in q.question for f in q.supporting_facts)


def test_generation_is_deterministic_and_seeded(world) -> None:
    a = generate_questions(world, QuestionSpec(n_questions=30, seed=1))
    assert a == generate_questions(world, QuestionSpec(n_questions=30, seed=1))
    assert a != generate_questions(world, QuestionSpec(n_questions=30, seed=2))
    assert len({q.id for q in a}) == len(a) == 30


def test_an_impossible_request_is_an_error() -> None:
    tiny = generate(make(n_entities=6, avg_degree=1, alias_rate=0.0))
    with pytest.raises(WorldGenerationError):
        generate_questions(tiny, QuestionSpec(n_questions=500, hop_mix={4: 1.0}))


@pytest.mark.parametrize("bad", [{}, {0: 1.0}, {5: 1.0}, {1: -1.0}, {1: 0.0}])
def test_invalid_hop_mix_is_rejected(bad: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        QuestionSpec(hop_mix=bad)


def test_cli_writes_questions(tmp_path) -> None:
    out = tmp_path / "q.json"
    code = main(
        [
            "questions",
            "generate",
            "--preset",
            "small",
            "--n",
            "12",
            "--hops",
            "1:1,2:1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text())
    assert len(data) == 12 and {q["hops"] for q in data} == {1, 2}
    assert all(q["supporting_facts"] for q in data)


# -- typed questions (ADR-0013) --------------------------------------------------------------------

MIXED = {"entity": 1.0, "set": 1.0, "count": 1.0, "compare": 1.0, "superlative": 1.0}


@pytest.fixture(scope="module")
def mixed(world):
    return generate_questions(world, QuestionSpec(n_questions=60, type_mix=MIXED))


def _final_set(world, q):
    """The answers of a set or count question, recomputed from the world's final facts."""
    facts = {e.id: e for e in world.edges}
    current = world.current_edges()
    frontier = {q.start}
    *lead, last_id = q.gold_paths[0]
    for fid in lead:
        e = facts[fid]
        forward = e.subject in frontier
        frontier = {
            (c.object if forward else c.subject)
            for c in current
            if c.predicate == e.predicate and (c.subject if forward else c.object) in frontier
        }
    (node,) = frontier
    e = facts[last_id]
    forward = e.subject == node
    return {
        (c.object if forward else c.subject)
        for c in current
        if c.predicate == e.predicate and (c.subject if forward else c.object) == node
    }


def test_the_default_spec_still_makes_only_entity_questions(world) -> None:
    qs = generate_questions(world, QuestionSpec(n_questions=20))
    assert {q.type for q in qs} == {"entity"} and all(q.answers == (q.answer,) for q in qs)


def test_type_mix_sets_how_many_of_each_type_are_made(mixed) -> None:
    by_type = Counter(q.type for q in mixed)
    assert len(mixed) == 60 and all(by_type[t] == 12 for t in ("set", "count", "compare"))
    # Few superlatives exist (one per relation and direction); plain questions make up the rest.
    assert 0 < by_type["superlative"] <= 12
    assert by_type["entity"] == 12 + 12 - by_type["superlative"]


def test_a_set_question_lists_exactly_the_entities_the_final_facts_reach(world, mixed) -> None:
    sets = [q for q in mixed if q.type == "set"]
    assert sets
    for q in sets:
        assert set(q.answers) == _final_set(world, q) and 2 <= len(q.answers) <= 25
        assert q.answer_kind == "set" and q.grading == "set_f1" and q.question.startswith("List")
        assert len(q.gold_paths) == len(q.answers) and q.supporting_facts == q.gold_paths[0]


def test_a_count_question_gold_is_the_size_of_the_set(world, mixed) -> None:
    counts = [q for q in mixed if q.type == "count"]
    assert counts
    for q in counts:
        assert q.number == len(_final_set(world, q)) == len(q.answers) >= 2
        assert q.answer_kind == "number" and q.answer_name == str(q.number)


def test_a_compare_question_gold_is_the_one_with_more_facts(world, mixed) -> None:
    facts = {e.id: e for e in world.edges}
    compares = [q for q in mixed if q.type == "compare"]
    assert compares
    for q in compares:
        (winner,) = q.answers
        by_entity: Counter[str] = Counter()
        for (fid,) in q.gold_paths:
            e = facts[fid]
            by_entity[e.subject if e.subject in (winner, q.start) else e.object] += 1
        assert len(set(by_entity.values())) == 2
        assert by_entity[winner] == max(by_entity.values())


def test_a_superlative_question_has_one_entity_with_the_most(world, mixed) -> None:
    supers = [q for q in mixed if q.type == "superlative"]
    assert supers and all(len(q.answers) == 1 and q.tie_rule for q in supers)
    assert all(q.question.startswith("Which") and "the most" in q.question for q in supers)


def test_a_dense_world_makes_up_missing_four_hop_questions_with_sets() -> None:
    dense = generate(make(n_entities=4000, avg_degree=12, degree_skew=1.0, seed=5))
    qs = generate_questions(dense, QuestionSpec(n_questions=10, hop_mix={4: 1.0}))
    assert len(qs) == 10 and Counter(q.type for q in qs)["set"] >= 1


def test_the_type_mix_is_validated() -> None:
    with pytest.raises(ValueError, match="type_mix"):
        QuestionSpec(type_mix={"riddle": 1.0})


def _ever(world):
    return {(e.subject, e.predicate, e.object) for e in world.edges}


def test_an_unanswerable_question_asks_for_a_fact_the_world_never_held(world) -> None:
    qs = generate_questions(world, QuestionSpec(n_questions=20, type_mix={"unanswerable": 1.0}))
    assert {q.type for q in qs} == {"unanswerable"}
    names = {e.id: e.name for e in world.entities}
    for q in qs:
        assert q.abstain == {"expected": True, "reason": "no_evidence"} and q.grading == "abstain"
        assert q.answers == () and q.supporting_facts == () and q.answer_kind == "abstain"
        asked = [
            (p, forward)
            for p, v in VERB.items()
            for forward, tail in (
                (True, f"{names[q.start]} {v}."),
                (False, f"{v} {names[q.start]}."),
            )
            if q.question.endswith(tail)
        ]
        assert len(asked) == 1
        predicate, forward = asked[0]
        assert not any(
            e.predicate == predicate and (e.subject if forward else e.object) == q.start
            for e in world.edges
        )


def test_a_false_premise_contradicts_exactly_one_true_fact(world) -> None:
    qs = generate_questions(world, QuestionSpec(n_questions=20, type_mix={"false_premise": 1.0}))
    facts = {e.id: e for e in world.edges}
    ever = _ever(world)
    assert len(qs) == 20
    for q in qs:
        assert q.abstain == {"expected": True, "reason": "false_premise"}
        asserted = q.premise["asserted_fact"]
        key = (asserted["subject"], asserted["predicate"], asserted["object"])
        assert key not in ever  # the premise is not a fact, current or past
        (fid,) = q.premise["contradicting_fact_ids"]
        true = facts[fid]
        assert true.predicate == asserted["predicate"] and q.start in (true.subject, true.object)
        assert true.id in {e.id for e in world.current_edges()}
        assert q.premise["correction"] in q.answer_name and q.gold_paths == ((fid,),)
        assert q.question.split(". ")[0].rstrip(".") not in q.premise["correction"]


def test_declining_types_fall_back_to_plain_questions_when_none_exist() -> None:
    tiny = generate(make(n_entities=40, avg_degree=2, seed=1))
    qs = generate_questions(tiny, QuestionSpec(n_questions=6, type_mix={"unanswerable": 1.0}))
    assert len(qs) == 6


def test_unanswerable_and_false_premise_avoid_facts_that_held_in_the_past() -> None:
    churny = generate(make(n_entities=80, avg_degree=6, churn_rate=0.7, seed=3))
    ever = _ever(churny)
    qs = generate_questions(
        churny, QuestionSpec(n_questions=30, type_mix={"unanswerable": 1.0, "false_premise": 1.0})
    )
    names = {e.id: e.name for e in churny.entities}
    for q in qs:
        if q.type == "false_premise":
            a = q.premise["asserted_fact"]
            assert (a["subject"], a["predicate"], a["object"]) not in ever
        elif q.type == "unanswerable":
            (predicate, forward) = next(
                (p, fw)
                for p, v in VERB.items()
                for fw, tail in (
                    (True, f"{names[q.start]} {v}."),
                    (False, f"{v} {names[q.start]}."),
                )
                if q.question.endswith(tail)
            )
            assert not any(
                e.predicate == predicate and (e.subject if forward else e.object) == q.start
                for e in churny.edges
            )
    assert {q.type for q in qs} == {"unanswerable", "false_premise"}


@pytest.fixture(scope="module")
def churned():
    return generate(
        make(n_entities=300, avg_degree=5, churn_rate=0.2, retraction_rate=0.1, alias_rate=0.3)
    )


def _versions(world, q):
    """Every fact of the question's subject, relation and kind of object, in time order."""
    ents = {e.id: e for e in world.entities}
    facts = {e.id: e for e in world.edges}
    gold = facts[q.gold_paths[0][0]]
    return [
        e
        for e in sorted(world.edges, key=lambda e: (e.valid_from, e.id))
        if e.subject == gold.subject
        and e.predicate == gold.predicate
        and ents[e.object].type == ents[gold.object].type
    ]


def test_an_as_of_question_gives_the_value_that_held_in_that_year(churned) -> None:
    qs = generate_questions(churned, QuestionSpec(n_questions=20, type_mix={"as_of": 1.0}))
    names = {e.id: e.name for e in churned.entities}
    assert {q.type for q in qs} == {"as_of"}
    for q in qs:
        held = {e.object for e in _versions(churned, q) if e in churned.edges_as_of(q.as_of)}
        assert held == set(q.answers) and len(held) == 1
        assert q.as_of is not None and str(q.as_of) in q.question
        assert q.stale_values and names[q.answers[0]] not in q.stale_values


def test_a_latest_question_has_the_current_value_and_lists_the_replaced_ones(churned) -> None:
    qs = generate_questions(churned, QuestionSpec(n_questions=20, type_mix={"latest": 1.0}))
    names = {e.id: e.name for e in churned.entities}
    for q in qs:
        versions = _versions(churned, q)
        now = {e.object for e in versions if e.valid_to is None}
        assert now == set(q.answers) and q.question.startswith("Currently")
        ended = {names[e.object] for e in versions if e.valid_to is not None}
        assert q.stale_values and set(q.stale_values) <= ended
        assert not set(q.stale_values) & {names[i] for i in now}


def test_a_retracted_question_has_no_current_answer(churned) -> None:
    qs = generate_questions(churned, QuestionSpec(n_questions=15, type_mix={"retracted": 1.0}))
    for q in qs:
        versions = _versions(churned, q)
        assert versions and all(e.valid_to is not None for e in versions)
        assert q.answers == () and q.abstain == {"expected": True, "reason": "retracted"}
        assert q.stale_values and q.answer_kind == "abstain"


def test_temporal_types_fall_back_when_the_world_never_changes(world) -> None:
    still = generate(make(n_entities=100, avg_degree=4, seed=4))
    qs = generate_questions(still, QuestionSpec(n_questions=5, type_mix={"as_of": 1.0}))
    assert len(qs) == 5 and {q.type for q in qs} == {"entity"}


def test_a_path_question_gold_is_every_shortest_chain_between_the_two(world) -> None:
    from kggen.spec import VAGUE_PREDICATE

    ents = {e.id: e for e in world.entities}
    facts = {e.id: e for e in world.edges}
    live = [
        e
        for e in world.current_edges()
        if e.predicate != VAGUE_PREDICATE
        and not ents[e.subject].distractor
        and not ents[e.object].distractor
    ]
    adj: dict[str, list[str]] = {}
    for e in live:
        adj.setdefault(e.subject, []).append(e.object)
        adj.setdefault(e.object, []).append(e.subject)
    qs = generate_questions(world, QuestionSpec(n_questions=30, type_mix={"path": 1.0}))
    assert qs and {q.type for q in qs} == {"path"}
    for q in qs:
        dist, ways = {q.start: 0}, {q.start: 1}  # shortest distance and number of shortest paths
        queue = [q.start]
        for node in queue:
            for other in adj.get(node, []):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    queue.append(other)
                if dist[other] == dist[node] + 1:
                    ways[other] = ways.get(other, 0) + (ways[node] if other in dist else 0)
        (end,) = q.answers
        assert dist[end] == q.hops and q.ordered and q.grading == "path"
        assert len(q.gold_paths) == ways[end]  # none of the shortest chains is left out
        assert len(q.gold_paths) == len(set(q.gold_paths))
        for chain in q.gold_paths:
            assert len(chain) == q.hops
            at = q.start
            for fid in chain:
                e = facts[fid]
                assert at in (e.subject, e.object)
                at = e.object if e.subject == at else e.subject
            assert at == end


def test_a_path_question_is_dropped_when_too_many_chains_link_the_pair() -> None:
    from kggen.questions import MAX_PATHS

    dense = generate(make(n_entities=300, avg_degree=14, seed=8))
    qs = generate_questions(dense, QuestionSpec(n_questions=40, type_mix={"path": 1.0}))
    assert qs and all(len(q.gold_paths) <= MAX_PATHS for q in qs)
