"""The kggen UI: pick a preset, set the knobs and a seed, generate a world, explore it, and browse
the multi-hop questions derived from it with each question's gold path on the graph."""

from __future__ import annotations

from dataclasses import dataclass

import indah

from kggen_ui import model

# The Graph component arrived in indah 0.12; an older indah still gets the form, stats, questions.
HAS_GRAPH = hasattr(indah, "Graph")


@dataclass
class UI:
    """The app's state and root component, kept apart so tests can drive it without a browser."""

    root: indah.Component
    preset: indah.Signal
    values: dict[str, indah.Signal]
    status: indah.Signal
    entities: indah.Signal
    facts: indah.Signal
    world: indah.Signal
    checks: indah.Signal
    graph: indah.Signal  # (nodes, edges) as the Graph component takes them
    graph_note: indah.Signal  # why there is no graph to draw, when there is not
    selected: indah.Signal  # the entity id the viewer clicked, or None
    window: indah.Signal  # how many nodes are drawn in full
    index: indah.Signal  # a model.WorldIndex for the detail panels
    question_values: dict[str, indah.Signal]  # the question form: how many, and the hop weights
    questions: indah.Signal  # the derived questions
    question_rows: indah.Signal  # the same, as table rows
    question_note: indah.Signal  # why there are no questions, when there are none
    picked: indah.Signal  # the id of the question the viewer chose, or None

    def current_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.values.items()}

    def current_question_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.question_values.items()}

    def picked_question(self):
        return model.find_question(self.questions.value, self.picked.value)

    def _derive(self, world) -> tuple[tuple, str]:
        """Questions for ``world`` from the current form, or none and the reason."""
        try:
            return model.derive_questions(world, self.current_question_values()), ""
        except ValueError as exc:
            return (), str(exc)

    def _set_questions(self, questions: tuple, note: str) -> None:
        self.picked.set(None)  # an old pick may name a question that is gone
        self.questions.set(questions)
        self.question_rows.set(model.question_rows(questions))
        self.question_note.set(note or (f"{len(questions)} questions." if questions else ""))

    def generate(self) -> None:
        try:
            world = model.generate_world(self.current_values())
        except ValueError as exc:
            self.status.set(str(exc))
            return
        summary = model.summarize(world)
        note = model.graph_limits_message(world)
        shown = ([], []) if note else model.graph_data(world)
        questions, question_note = self._derive(world)

        def apply() -> None:
            # The old selection and pick name things that may not exist in the new world, so they
            # go first; one batch means the graph never sees the new data with the old ones.
            self.selected.set(None)
            self.world.set(world)
            self.index.set(model.index_world(world))
            self.graph.set(shown)
            self.graph_note.set(note or "")
            self._set_questions(questions, question_note)
            self.checks.set(model.check_rows(world))
            self.entities.set(f"{summary.entities:,}")
            self.facts.set(f"{summary.facts:,}")
            self.status.set(f"Generated with seed {summary.seed}.")

        indah.batch(apply)

    def derive(self) -> None:
        """Derive a new set of questions from the current world, with the current form."""
        world = self.world.value
        if world is None:
            self.question_note.set("Generate a world first.")
            return
        questions, note = self._derive(world)
        indah.batch(lambda: self._set_questions(questions, note))

    def highlight(self) -> dict | None:
        """What the graph emphasises: the picked question's entities and facts, if it is drawn."""
        if self.graph_note.value:
            return None
        return model.highlight_for(self.index.value, self.picked_question())

    def focus(self) -> str | None:
        """Where the graph centres: the picked question's start entity."""
        if self.graph_note.value:
            return None
        question = self.picked_question()
        return question.start if question else None


def build_ui() -> UI:
    """Build one viewer's UI: independent signals, so two tabs never share a world."""
    start = model.preset_values(model.DEFAULT_PRESET)
    preset = indah.Signal(model.DEFAULT_PRESET)
    values = {name: indah.Signal(value) for name, value in start.items()}
    status = indah.Signal("Choose a preset or set the knobs, then generate.")

    def apply_preset() -> None:
        for name, value in model.preset_values(preset.value).items():
            values[name].set(value)

    indah.effect(apply_preset)  # also runs once now, which fills the form from the first preset

    ui = UI(
        root=None,  # type: ignore[arg-type]
        preset=preset,
        values=values,
        status=status,
        entities=indah.Signal("-"),
        facts=indah.Signal("-"),
        world=indah.Signal(None),
        checks=indah.Signal([]),
        graph=indah.Signal(([], [])),
        graph_note=indah.Signal(""),
        selected=indah.Signal(None),
        window=indah.Signal(300),
        index=indah.Signal(None),
        question_values={k: indah.Signal(v) for k, v in model.QUESTION_DEFAULTS.items()},
        questions=indah.Signal(()),
        question_rows=indah.Signal([]),
        question_note=indah.Signal(""),
        picked=indah.Signal(None),
    )

    def group_inputs(group: str) -> list[indah.Component]:
        return [
            indah.Number(values[k.name], min=k.min, max=k.max, step=k.step, label=k.label)
            for k in model.KNOBS
            if k.group == group
        ]

    def verdict() -> str:
        rows = ui.checks.value
        if not rows:
            return "Generate a world to compare what it realized against what you asked for."
        off = [r[0] for r in rows if r[3] != "ok"]
        return "All knobs are within tolerance." if not off else "Off target: " + ", ".join(off)

    ui.root = indah.Column(
        [
            indah.Card(
                title="Generate a world",
                children=[
                    indah.Select(preset, options=list(model.PRESET_NAMES), label="Preset"),
                    *[
                        indah.Column([indah.Text(title), indah.Grid(group_inputs(group))])
                        for group, title in model.GROUPS.items()
                    ],
                    indah.Button("Generate", ui.generate),
                    indah.Text(lambda: ui.status.value),
                ],
            ),
            indah.Row(
                [
                    indah.Stat(lambda: ui.entities.value, label="Entities"),
                    indah.Stat(lambda: ui.facts.value, label="Facts"),
                ]
            ),
            world_card(ui),
            questions_card(ui),
            *entity_cards(ui),
            indah.Card(
                title="Asked for vs. realized",
                children=[
                    indah.Text(verdict),
                    indah.DataFrame(
                        lambda: {
                            "columns": ["Knob", "Asked for", "Realized", "Result"],
                            "rows": ui.checks.value,
                        }
                    ),
                ],
            ),
        ]
    )
    return ui


def world_card(ui: UI) -> indah.Component:
    """The graph, or a note when this indah has no Graph yet."""
    if not HAS_GRAPH:
        return indah.Card(
            title="The world",
            children=[indah.Text("The graph view needs indah 0.12 or newer; yours has no Graph.")],
        )
    graph = indah.Graph(
        lambda: ui.graph.value[0],
        lambda: ui.graph.value[1],
        selected=ui.selected,
        focus_size=ui.window,
        highlight=ui.highlight,
        focus=ui.focus,
        height=520,
        max_nodes=model.MAX_GRAPH_NODES,
        label="The generated world: entities coloured by type",
    )
    return indah.Card(
        title="The world",
        children=[
            indah.Text(lambda: ui.graph_note.value or ""),
            graph,
            indah.Slider(ui.window, min=20, max=2000, step=10, label="Nodes drawn in full"),
        ],
    )


def questions_card(ui: UI) -> indah.Component:
    """The multi-hop questions, a control for their hop mix, and the picked one's gold path."""
    inputs = [
        indah.Number(ui.question_values[k.name], min=k.min, max=k.max, step=k.step, label=k.label)
        for k in model.QUESTION_KNOBS
    ]
    table = indah.Table(
        rows=ui.question_rows,
        columns=[
            indah.Col("id", "ID", kind="mono"),
            indah.Col("hops", "Hops", kind="number"),
            indah.Col("question", "Question", wrap=True),
            indah.Col("answer", "Answer"),
        ],
        key="id",
        selected=ui.picked,
        height="300px",
        label="Questions",
    )
    return indah.Card(
        title="Questions",
        children=[
            indah.Text(
                "Questions come from the world, never from a pipeline's output. Pick one to "
                "see its gold path on the graph."
            ),
            indah.Grid(inputs),
            indah.Button("Derive questions", ui.derive, variant="tonal"),
            indah.Text(lambda: ui.question_note.value),
            table,
            indah.Text(lambda: model.answer_text(ui.picked_question())),
            indah.DataFrame(
                lambda: {
                    "columns": ["Hop", "Fact", "Reads", "Held"],
                    "rows": model.gold_rows(ui.index.value, ui.picked_question()),
                }
            ),
        ],
    )


def entity_cards(ui: UI) -> list[indah.Component]:
    """The detail of the entity clicked on the graph (there is no click without a Graph)."""
    if not HAS_GRAPH:
        return []
    return [
        indah.Card(
            title="Selected entity",
            children=[
                indah.Text(lambda: model.entity_summary(ui.index.value, ui.selected.value)),
                indah.DataFrame(
                    lambda: {
                        "columns": model.FACT_COLUMNS,
                        "rows": model.fact_rows(ui.index.value, ui.selected.value),
                    }
                ),
            ],
        )
    ]


def build_session() -> indah.Session:
    return indah.Session(build_ui().root)


def serve(port: int | None = None, host: str = "127.0.0.1") -> None:
    """Start the app; blocks until stopped."""
    indah.launch(
        indah.create_app(session_factory=build_session, title="kggen"), port=port, host=host
    )
