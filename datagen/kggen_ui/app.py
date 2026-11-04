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
    doc_values: dict[str, indah.Signal]  # the render form: alias rate, facts per document
    documents: indah.Signal  # the rendered documents
    doc_rows: indah.Signal  # the same, as table rows
    doc_note: indah.Signal  # a note about the documents (how many, or why none)
    llm_on: indah.Signal  # the opt-in switch: Render documents may call the LLM (off by default)
    llm_endpoint: indah.Signal  # an OpenAI-compatible base URL; blank means the cache only
    llm_model: indah.Signal  # the model name sent to the endpoint and used as the cache key
    doc_source: indah.Signal  # "templates" or "llm": who wrote the documents on screen
    sentences: indah.Signal  # fact id -> the sentence that expresses it
    picked_doc: indah.Signal  # the id of the document being read, or None
    picked_fact: indah.Signal  # the fact id of the sentence the viewer clicked, or None
    question_spec: indah.Signal  # the QuestionSpec the questions on screen were derived with
    render_spec: indah.Signal  # the RenderSpec the documents on screen were rendered with
    export_file: indah.Signal  # a DownloadFile once the export is prepared, else None
    export_note: indah.Signal  # how big the prepared export is, or why there is none

    def current_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.values.items()}

    def current_question_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.question_values.items()}

    def current_doc_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.doc_values.items()}

    def picked_question(self):
        return model.find_question(self.questions.value, self.picked.value)

    def picked_document(self):
        return model.find_document(self.documents.value, self.picked_doc.value)

    def _render(self, world, llm: bool = False) -> tuple[tuple, str]:
        """Documents for ``world`` from the current form, or none and the reason.

        Templates unless ``llm``; only the Render button passes it, so generating a world never
        makes a provider call.
        """
        try:
            if llm:
                rendered = model.render_world_llm(
                    world,
                    self.current_doc_values(),
                    self.llm_endpoint.value,
                    self.llm_model.value,
                )
                return rendered.documents, model.llm_note(rendered, self.llm_model.value)
            documents = model.render_world(world, self.current_doc_values())
        except ValueError as exc:
            return (), str(exc)
        shown = min(len(documents), model.MAX_DOCUMENT_ROWS)
        more = f" (the first {shown:,} are listed)" if shown < len(documents) else ""
        return documents, f"{len(documents):,} documents{more}."

    def _set_documents(self, documents: tuple, note: str, source: str = "templates") -> None:
        self._drop_export()
        self.doc_source.set(source)
        try:
            self.render_spec.set(model.render_spec_from_values(self.current_doc_values()))
        except ValueError:
            self.render_spec.set(None)
        self.picked_doc.set(None)  # old picks may name documents and sentences that are gone
        self.picked_fact.set(None)
        self.documents.set(documents)
        self.doc_rows.set(model.document_rows(documents))
        self.sentences.set(model.sentence_lookup(documents))
        self.doc_note.set(note)

    def _derive(self, world) -> tuple[tuple, str]:
        """Questions for ``world`` from the current form, or none and the reason."""
        try:
            return model.derive_questions(world, self.current_question_values()), ""
        except ValueError as exc:
            return (), str(exc)

    def _drop_export(self) -> None:
        """An export holds the world, questions and documents, so any change makes it stale."""
        self.export_file.set(None)
        self.export_note.set("")

    def _set_questions(self, questions: tuple, note: str) -> None:
        self._drop_export()
        try:
            self.question_spec.set(model.question_spec_from_values(self.current_question_values()))
        except ValueError:
            self.question_spec.set(None)
        self.picked.set(None)  # an old pick may name a question that is gone
        self.questions.set(questions)
        self.question_rows.set(model.question_rows(questions))
        self.question_note.set(note or model.question_summary(questions))

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
        documents, doc_note = self._render(world)

        def apply() -> None:
            # The old selection and pick name things that may not exist in the new world, so they
            # go first; one batch means the graph never sees the new data with the old ones.
            self.selected.set(None)
            self.world.set(world)
            self.index.set(model.index_world(world))
            self.graph.set(shown)
            self.graph_note.set(note or "")
            self._set_questions(questions, question_note)
            self._set_documents(documents, doc_note)
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

    def render(self) -> None:
        """Render the current world's facts as documents again, with the current form."""
        world = self.world.value
        if world is None:
            self.doc_note.set("Generate a world first.")
            return
        llm = bool(self.llm_on.value)
        documents, note = self._render(world, llm)
        if llm and not documents:  # a failed LLM render must not wipe the documents on screen
            self.doc_note.set(note)
            return
        source = "llm" if llm else "templates"
        indah.batch(lambda: self._set_documents(documents, note, source))

    def prepare_export(self) -> None:
        """Build the dataset zip for the world on screen and offer it for download."""
        world = self.world.value
        if world is None:
            self.export_note.set("Generate a world first.")
            return
        data, name = model.export_dataset(world, self.question_spec.value, self.render_spec.value)
        self.export_file.set(indah.DownloadFile(data, name, "application/zip"))
        note = model.export_summary(data, name)
        if self.doc_source.value == "llm":
            note += " Its documents.json is template text: the LLM-written set is not exported."
        self.export_note.set(note)

    def highlight(self) -> dict | None:
        """What the graph emphasises: the clicked sentence's fact, else the picked question's path.
        Nothing when the world is too big to draw."""
        if self.graph_note.value:
            return None
        fact = model.fact_highlight(self.index.value, self.picked_fact.value)
        return fact or model.highlight_for(self.index.value, self.picked_question())

    def focus(self) -> str | None:
        """Where the graph centres: the clicked fact's subject, else the picked question's start."""
        if self.graph_note.value:
            return None
        index = self.index.value
        if index is not None and self.picked_fact.value in index.facts:
            return index.facts[self.picked_fact.value].subject
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
        doc_values={k: indah.Signal(v) for k, v in model.RENDER_DEFAULTS.items()},
        documents=indah.Signal(()),
        doc_rows=indah.Signal([]),
        doc_note=indah.Signal(""),
        llm_on=indah.Signal(False),
        llm_endpoint=indah.Signal(""),
        llm_model=indah.Signal(model.LLM_DEFAULT_MODEL),
        doc_source=indah.Signal("templates"),
        sentences=indah.Signal({}),
        picked_doc=indah.Signal(None),
        picked_fact=indah.Signal(None),
        question_spec=indah.Signal(None),
        render_spec=indah.Signal(None),
        export_file=indah.Signal(None),
        export_note=indah.Signal(""),
    )

    # One thing is emphasised on the graph at a time: picking a question clears the clicked
    # sentence and the reverse, and opening another document clears its sentence. Each effect acts
    # only on a real pick, so they cannot undo each other.
    def question_picked() -> None:
        if ui.picked.value is not None:
            ui.picked_fact.set(None)

    def fact_picked() -> None:
        if ui.picked_fact.value is not None:
            ui.picked.set(None)

    def document_opened() -> None:
        ui.picked_doc.value  # noqa: B018 (read to subscribe)
        ui.picked_fact.set(None)

    for effect in (question_picked, fact_picked, document_opened):
        indah.effect(effect)

    def group_inputs(group: str) -> list[indah.Component]:
        return [field(k, values[k.name]) for k in model.KNOBS if k.group == group]

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
            documents_card(ui),
            export_card(ui),
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


def field(knob: model.Knob, signal: indah.Signal) -> indah.Component:
    """A number field with an info icon that says what the setting controls."""
    number = indah.Number(signal, min=knob.min, max=knob.max, step=knob.step, label=knob.label)
    tip = model.knob_help(knob)
    if not tip:
        return number
    return indah.Row([number, indah.InfoTip(tip, label=f"About {knob.label}")])


def questions_card(ui: UI) -> indah.Component:
    """The multi-hop questions, a control for their hop mix, and the picked one's gold path."""
    inputs = [field(k, ui.question_values[k.name]) for k in model.QUESTION_KNOBS]
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


def documents_card(ui: UI) -> indah.Component:
    """Documents rendered from the world's facts, and the sentences of the one being read."""
    inputs = [field(k, ui.doc_values[k.name]) for k in model.RENDER_KNOBS]
    documents = indah.Table(
        rows=ui.doc_rows,
        columns=[
            indah.Col("id", "ID", kind="mono"),
            indah.Col("title", "About"),
            indah.Col("sentences", "Sentences", kind="number"),
            indah.Col("kind", "Kind"),
        ],
        key="id",
        selected=ui.picked_doc,
        height="260px",
        search=True,
        label="Documents",
    )
    sentence_rows = indah.computed(lambda: model.sentence_rows(ui.picked_document()))
    sentences = indah.Table(
        rows=sentence_rows,
        columns=[
            indah.Col("fact", "Fact", kind="mono"),
            indah.Col("sentence", "Sentence", wrap=True),
        ],
        key="fact",
        selected=ui.picked_fact,
        label="Sentences (click one)",
    )
    return indah.Card(
        title="Documents",
        children=[
            indah.Text(
                "Documents are written from the world's facts by templates: deterministic, no "
                "network, no LLM. Every sentence knows which fact it expresses."
            ),
            indah.Grid(inputs),
            indah.Switch(
                ui.llm_on, label="Write with an LLM instead (opt-in; may call a provider)"
            ),
            indah.Text(
                "Only Render documents uses it, never Generate. Answers are cached in "
                f"{model.LLM_CACHE_DIR}, so the same world again calls nothing. A blank endpoint "
                "uses the cache only. The key is read from KGGEN_LLM_API_KEY."
            ),
            indah.Grid(
                [
                    indah.TextInput(
                        ui.llm_endpoint,
                        label="Endpoint (OpenAI-compatible base URL)",
                        placeholder="http://127.0.0.1:8080/v1",
                    ),
                    indah.TextInput(ui.llm_model, label="Model"),
                ]
            ),
            indah.Button("Render documents", ui.render, variant="tonal"),
            indah.Text(lambda: ui.doc_note.value),
            documents,
            indah.Text(lambda: doc_heading(ui)),
            sentences,
            indah.Text(lambda: model.fact_note(ui.index.value, ui.picked_fact.value)),
            indah.DataFrame(
                lambda: {
                    "columns": ["Written as", "Entity", "Id"],
                    "rows": model.mention_rows(
                        ui.index.value, ui.sentences.value, ui.picked_fact.value
                    ),
                }
            ),
        ],
    )


def export_card(ui: UI) -> indah.Component:
    """Download the world, questions and documents as one dataset that can be made again."""
    return indah.Card(
        title="Export",
        children=[
            indah.Text(
                "One zip with the world, the questions with their gold paths, the documents, "
                "and the specs that made them. Run the command in its README on those specs and "
                "you get the same bytes back."
            ),
            indah.Button("Prepare export", ui.prepare_export, variant="tonal"),
            indah.Text(lambda: ui.export_note.value),
            indah.Download(ui.export_file, label="Download the dataset (.zip)"),
        ],
    )


def doc_heading(ui: UI) -> str:
    document = ui.picked_document()
    if document is None:
        return "Pick a document to read it."
    kind = (
        " (a distractor document: it is about entities no question uses)"
        if document.distractor
        else ""
    )
    return f"{document.id}: about {document.title}{kind}"


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
