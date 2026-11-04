"""Filler prose for long documents: sentences that carry no facts and name no entity.

A realistic 500- or 5,000-word document is mostly background, so the facts a question needs are a
small part of the text. These sentences are built from frames and word banks, with an RNG the
caller owns, so the same seed gives the same text. A frame may name the document's subject
(``{S}``); the renderer records that as a mention with no fact. Nothing else in a frame is a name.
"""

from __future__ import annotations

import random
import re

# What each kind of entity is called when a sentence refers back to it.
NOMINAL = {
    "PERSON": "this person",
    "ORG": "the organization",
    "LOCATION": "the area",
    "EVENT": "the event",
    "PRODUCT": "the product",
    "NORP": "the community",
    "LAW": "the statute",
}

BANKS: dict[str, tuple[str, ...]] = {
    "adj": (
        "steady",
        "considerable",
        "quiet",
        "growing",
        "unusual",
        "lasting",
        "mixed",
        "cautious",
        "broad",
        "local",
        "close",
        "persistent",
        "measured",
        "uneven",
    ),
    "obs": (
        "commentators",
        "neighbours",
        "analysts",
        "reviewers",
        "colleagues",
        "observers",
        "regional press",
        "longtime partners",
        "independent researchers",
        "newcomers",
    ),
    "q1": (
        "careful planning",
        "its patient approach",
        "a habit of keeping records",
        "slow, steady growth",
        "plain dealing",
        "attention to detail",
        "a preference for small steps",
        "reliable follow-through",
    ),
    "q2": (
        "a willingness to revise earlier plans",
        "its modest public profile",
        "consistent standards",
        "an unwillingness to overpromise",
        "cooperation across groups",
        "a long memory for past mistakes",
    ),
    "theme": (
        "how decisions were made",
        "the question of continuity",
        "the cost of expansion",
        "who carried the day-to-day work",
        "the balance between caution and ambition",
        "the role of timing",
        "the habit of documenting everything",
        "what changed after the early period",
    ),
    "contrast": (
        "others saw it differently",
        "the details were disputed",
        "the record is incomplete",
        "not everyone agreed at the time",
        "later accounts soften the picture",
        "the evidence points in more than one direction",
    ),
    "time": (
        "in the early period",
        "during the middle years",
        "in more recent years",
        "for most of that time",
        "after the first reorganization",
        "before the later changes",
        "through several seasons",
    ),
    "verb": (
        "shaped",
        "complicated",
        "eased",
        "redirected",
        "reinforced",
        "tested",
        "clarified",
        "slowed",
    ),
    "thing": (
        "routine",
        "planning",
        "reporting",
        "outreach",
        "scheduling",
        "record keeping",
        "training",
        "budgeting",
        "coordination",
        "review",
    ),
    "result": (
        "fewer surprises",
        "slower but safer progress",
        "a clearer division of work",
        "more frequent check-ins",
        "a longer list of open questions",
        "better notes for whoever came next",
    ),
}

FRAMES: dict[str, tuple[str, ...]] = {
    "history": (
        "In its early years, {it} drew {adj} attention from {obs}.",
        "Over time, {it} became known for {q1} and {q2}.",
        "Accounts of the period describe {it} as {adj}, though {contrast}.",
        "{time}, {it} faced questions about {theme}.",
        "Few discussions of {S} leave out {theme}.",
        "{time}, the story of {it} was told mostly by {obs}.",
        "Looking back, {obs} tend to describe {it} as {adj}.",
        "What {obs} remember most is {q1}, which {verb} much of what followed.",
    ),
    "operations": (
        "Day to day, {thing} {verb} how {it} worked, and {contrast}.",
        "Much of the work came down to {thing} and {thing}, which produced {result}.",
        "{time}, {thing} took more time than expected, leading to {result}.",
        "{It} relied on {thing} more than on {thing}, a choice that {verb} later decisions.",
        "{obs} often pointed to {q1} as the reason {thing} went smoothly.",
        "Because of {q2}, {thing} rarely changed from one year to the next.",
        "Written procedures for {thing} existed, but in practice {q1} mattered more.",
        "The effect on {thing} was {adj}, and it left {result}.",
    ),
    "community": (
        "Those close to {it} speak of {q1} and of {theme}.",
        "{obs} sometimes disagreed about {theme}, and the discussion continued for years.",
        "Public reaction was {adj}, with {obs} divided over {theme}.",
        "{time}, the conversation turned to {theme}, and {contrast}.",
        "Interest in {S} was {adj} but never faded entirely.",
        "{It} was often discussed alongside {theme}.",
        "A {adj} number of people took an interest in {thing}.",
        "Opinions differed on whether {q2} was a strength or a weakness.",
    ),
    "outlook": (
        "Going forward, {obs} expect {thing} to remain {adj}.",
        "Whether {q1} will last is an open question, and {contrast}.",
        "Plans for {thing} are {adj}, and they depend on {theme}.",
        "The next few years will test {q2} as much as {q1}.",
        "Little suggests that {theme} will matter less in the future.",
        "{obs} caution that {thing} could bring {result}.",
        "In summary, {it} is best understood through {q1} and {theme}.",
        "What happens next depends largely on {thing} and on {theme}.",
    ),
}
TOPICS = tuple(FRAMES)


def filler_sentence(
    rng: random.Random,
    entity_type: str,
    topic: str | None = None,
    avoid: set[str] | None = None,
) -> str:
    """One filler sentence. ``{S}`` is left in place for the renderer to fill with the subject.

    Each slot is drawn on its own, so a frame that uses ``{thing}`` twice gets two different words.
    A sentence already in ``avoid`` is drawn again (a few times) and the result is added to it, so
    one document does not repeat a sentence word for word.
    """
    nominal = NOMINAL.get(entity_type, "the subject")

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "S":
            return "{S}"
        if name == "it":
            return nominal
        if name == "It":
            return nominal[0].upper() + nominal[1:]
        return rng.choice(BANKS[name])

    for _ in range(30):
        frame = rng.choice(FRAMES[topic or rng.choice(TOPICS)])
        text = re.sub(r"\{(\w+)\}", fill, frame)
        text = text[0].upper() + text[1:]
        if avoid is None or text not in avoid:
            break
    if avoid is not None:
        avoid.add(text)
    return text
