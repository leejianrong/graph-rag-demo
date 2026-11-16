"""The kggen UI: pick a preset, set the knobs and a seed, generate a world, explore it, and browse
the multi-hop questions derived from it with each question's gold path on the graph."""

from __future__ import annotations

from dataclasses import dataclass

import indah

from kggen_ui import model, workspaces

# The Graph component arrived in indah 0.12; an older indah still gets the form, stats, questions.
HAS_GRAPH = hasattr(indah, "Graph")


YEAR_VIEW_WINDOW = 1000
YEAR_LEGEND = (
    "Green: new this year. Purple: renamed this year. Rust: ends this year. Pale: not in this "
    "year. Everything else holds."
)


@dataclass
class UI:
    """The app's state and root component, kept apart so tests can drive it without a browser."""

    root: indah.Component
    preset: indah.Signal
    difficulty: indah.Signal
    evolution: indah.Signal
    persona: indah.Signal
    timeline: indah.Signal
    phrasing: indah.Signal
    name_style: indah.Signal
    values: dict[str, indah.Signal]
    status: indah.Signal
    entities: indah.Signal
    facts: indah.Signal
    world: indah.Signal
    checks: indah.Signal
    graph: indah.Signal  # (nodes, edges) as the Graph component takes them
    graph_note: indah.Signal  # why there is no graph to draw, when there is not
    tgraph: indah.Signal  # a model.TimeGraph when the world changes over time, else None
    time_on: indah.Signal  # draw the world as of a year instead of its final state
    offset: indah.Signal  # years after the first year: the year being shown when time_on
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
    store: workspaces.Store  # saved workspaces on disk
    ws_rows: indah.Signal  # the saved workspaces, as table rows
    ws_current: indah.Signal  # the slug of the open workspace, or None (a scratch session)
    ws_picked: indah.Signal  # the slug picked in the rail
    ws_name: indah.Signal  # the name typed for a new workspace
    ws_note: indah.Signal  # what the last workspace action did, or why it failed
    rev_rows: indah.Signal  # the open workspace's revisions, newest first
    rev_picked: indah.Signal  # the revision id picked in History
    ws_confirm: indah.Signal  # whether the delete confirmation is open
    step: indah.Signal  # the wizard step shown (an index into model.WIZARD_STEPS)
    tab: indah.Signal  # the tab shown (an index; 0 is Setup, 1 is World)

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
            documents = model.render_world(
                world, self.current_doc_values(), self.phrasing.value, self.timeline.value == "on"
            )
        except ValueError as exc:
            return (), str(exc)
        shown = min(len(documents), model.MAX_DOCUMENT_ROWS)
        more = f" (the first {shown:,} are listed)" if shown < len(documents) else ""
        return documents, f"{len(documents):,} documents{more}."

    def _set_documents(self, documents: tuple, note: str, source: str = "templates") -> None:
        self._drop_export()
        self.doc_source.set(source)
        try:
            self.render_spec.set(
                model.render_spec_from_values(
                    self.current_doc_values(), self.phrasing.value, self.timeline.value == "on"
                )
            )
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
            return (
                model.derive_questions(
                    world, self.current_question_values(), self.phrasing.value, self.persona.value
                ),
                "",
            )
        except ValueError as exc:
            return (), str(exc)

    def _drop_export(self) -> None:
        """An export holds the world, questions and documents, so any change makes it stale."""
        self.export_file.set(None)
        self.export_note.set("")

    def _set_questions(self, questions: tuple, note: str) -> None:
        self._drop_export()
        try:
            self.question_spec.set(
                model.question_spec_from_values(
                    self.current_question_values(), self.phrasing.value, self.persona.value
                )
            )
        except ValueError:
            self.question_spec.set(None)
        self.picked.set(None)  # an old pick may name a question that is gone
        self.questions.set(questions)
        self.question_rows.set(model.question_rows(questions))
        self.question_note.set(note or model.question_summary(questions))

    def generate(self) -> None:
        try:
            world = model.generate_world(self.current_values(), name_style=self.name_style.value)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        summary = model.summarize(world)
        nodes, edges, note = model.graph_view(world)
        shown = (nodes, edges)
        tgraph = model.time_graph(world)
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
            self.tgraph.set(tgraph)
            self.time_on.set(False)  # a new world opens on its final state
            self.offset.set(0)
            self._set_questions(questions, question_note)
            self._set_documents(documents, doc_note)
            self.checks.set(model.check_rows(world))
            self.entities.set(f"{summary.entities:,}")
            self.facts.set(f"{summary.facts:,}")
            self.status.set(f"Generated with seed {summary.seed}.")

        indah.batch(apply)
        self._save_revision()

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

    # -- workspaces ---------------------------------------------------------------------------

    def settings(self) -> dict:
        """Everything on the form: what a workspace saves, and all that regenerates its data."""
        return {
            "preset": self.preset.value,
            "difficulty": self.difficulty.value,
            "evolution": self.evolution.value,
            "persona": self.persona.value,
            "timeline": self.timeline.value,
            "phrasing": self.phrasing.value,
            "name_style": self.name_style.value,
            "values": self.current_values(),
            "question_values": self.current_question_values(),
            "doc_values": self.current_doc_values(),
        }

    def apply_settings(self, saved: dict) -> None:
        """Put saved settings back on the form. The selects go first, one by one, because each
        refills the numbers from its preset; the saved numbers go on top afterwards."""
        for name in ("preset", "difficulty", "evolution", "persona"):
            if name in saved:
                getattr(self, name).set(saved[name])
        for name in ("timeline", "phrasing", "name_style"):
            if name in saved:
                getattr(self, name).set(saved[name])
        for group, signals in (
            ("values", self.values),
            ("question_values", self.question_values),
            ("doc_values", self.doc_values),
        ):
            for key, value in (saved.get(group) or {}).items():
                if key in signals:
                    signals[key].set(value)

    def refresh_workspaces(self) -> None:
        self.ws_rows.set(
            [
                {"slug": w.slug, "name": w.name, "saved": (w.latest.saved if w.latest else "")}
                for w in self.store.list()
            ]
        )
        self.refresh_revisions()

    def refresh_revisions(self) -> None:
        slug = self.ws_current.value
        try:
            ws = self.store.load(slug) if slug else None
        except workspaces.WorkspaceError:
            ws = None
        self.rev_rows.set(
            [
                {"id": str(r.id), "saved": r.saved, "what": model.settings_summary(r.settings)}
                for r in reversed(ws.revisions)
            ]
            if ws
            else []
        )
        self.rev_picked.set(None)

    def workspace_title(self) -> str:
        slug = self.ws_current.value
        if not slug:
            return (
                "Scratch session: not saved. Name a workspace in the rail to keep these settings."
            )
        try:
            return self.store.load(slug).name
        except workspaces.WorkspaceError:
            return slug

    def _ws_do(self, action) -> None:  # type: ignore[no-untyped-def]
        """Run a store action; a failure becomes the note instead of a crash."""
        try:
            action()
        except workspaces.WorkspaceError as exc:
            self.ws_note.set(str(exc))

    def create_workspace(self) -> None:
        """A new workspace holding the form as it is now; it becomes the open one."""

        def run() -> None:
            ws = self.store.create(self.ws_name.value, self.settings())
            self.ws_current.set(ws.slug)
            self.ws_picked.set(ws.slug)
            self.ws_name.set("")
            self.refresh_workspaces()
            self.ws_note.set(f"Created {ws.name}.")

        self._ws_do(run)

    def open_workspace(self, slug: str | None) -> None:
        """Open a saved workspace: its latest settings go on the form and the data is made again
        from them (not for a stress or large preset, which takes long enough to start by hand)."""
        if not slug or slug == self.ws_current.value:
            return

        def run() -> None:
            ws = self.store.load(slug)
            self.ws_current.set(slug)
            if ws.latest is not None:
                self.apply_settings(ws.latest.settings)
                if self.preset.value in ("tiny", "small", "medium"):
                    self.generate()
                else:
                    self.status.set("Settings restored. Press Generate to make the data.")
            self.refresh_revisions()
            self.ws_note.set(f"Opened {ws.name}.")

        self._ws_do(run)

    def restore_revision(self) -> None:
        """Put the picked revision's settings back and generate; that saves a new revision."""
        slug, picked = self.ws_current.value, self.rev_picked.value
        if not slug or picked is None:
            self.ws_note.set("Pick a revision first.")
            return

        def run() -> None:
            ws = self.store.load(slug)
            revision = next(r for r in ws.revisions if str(r.id) == str(picked))
            self.apply_settings(revision.settings)
            self.generate()
            self.ws_note.set(f"Restored revision {revision.id}.")

        self._ws_do(run)

    def rename_workspace(self) -> None:
        slug = self.ws_current.value
        if not slug:
            self.ws_note.set("Open a workspace first.")
            return

        def run() -> None:
            ws = self.store.rename(slug, self.ws_name.value)
            self.ws_name.set("")
            self.refresh_workspaces()
            self.ws_note.set(f"Renamed to {ws.name}.")

        self._ws_do(run)

    def duplicate_workspace(self) -> None:
        slug = self.ws_current.value
        if not slug:
            self.ws_note.set("Open a workspace first.")
            return

        def run() -> None:
            name = self.ws_name.value or f"{self.store.load(slug).name} copy"
            ws = self.store.duplicate(slug, name)
            self.ws_name.set("")
            self.ws_current.set(ws.slug)
            self.ws_picked.set(ws.slug)
            self.refresh_workspaces()
            self.ws_note.set(f"Copied the latest settings to {ws.name}.")

        self._ws_do(run)

    def delete_workspace(self) -> None:
        slug = self.ws_current.value
        self.ws_confirm.set(False)
        if not slug:
            return

        def run() -> None:
            self.store.delete(slug)
            self.ws_current.set(None)
            self.ws_picked.set(None)
            self.refresh_workspaces()
            self.ws_note.set("Deleted. The form keeps its settings as a scratch session.")

        self._ws_do(run)

    def go(self, step: int) -> None:
        """Show a wizard step (clamped to the steps there are)."""
        self.step.set(max(0, min(len(model.WIZARD_STEPS) - 1, step)))

    def generate_and_show(self) -> None:
        """Generate from the Review step, then open the World tab if there is a world to see."""
        self.generate()
        if self.world.value is not None:
            self.tab.set(1)

    def step_items(self) -> list[dict]:
        at = self.step.value
        return [
            {"label": name, "state": "done" if i < at else "current" if i == at else "pending"}
            for i, name in enumerate(model.WIZARD_STEPS)
        ]

    def _save_revision(self) -> None:
        """After a successful Generate in an open workspace: keep the settings that made it."""
        slug = self.ws_current.value
        if slug:
            self._ws_do(lambda: self.store.save(slug, self.settings()))
            self.refresh_workspaces()

    def viewing_time(self) -> bool:
        """Whether the graph shows a year: the switch is on and the world has years to show."""
        return bool(self.time_on.value) and self.tgraph.value is not None

    def drawn(self) -> tuple[list[dict], list[dict]]:
        """The nodes and edges on the graph: the final state, or the world as of the chosen year."""
        if self.viewing_time():
            return self.tgraph.value.nodes_edges(self.current_year())
        return self.graph.value

    def drawn_in_full(self) -> int:
        """How many nodes the graph draws in full. The year view draws at least 1,000: a node
        outside the window is a dot in its type's colour, which would hide what the year did."""
        return (
            max(int(self.window.value), YEAR_VIEW_WINDOW)
            if self.viewing_time()
            else int(self.window.value)
        )

    def current_year(self) -> int:
        """The year the slider is on (the first year when the world has no years to step)."""
        tgraph = self.tgraph.value
        if tgraph is None:
            return 0
        return tgraph.clamp(tgraph.first + int(self.offset.value))

    def step_year(self, direction: int, by_change: bool) -> None:
        """Move the year one year, or to the next year anything changes."""
        tgraph = self.tgraph.value
        if tgraph is not None:
            year = tgraph.step(self.current_year(), direction, by_change)
            self.offset.set(year - tgraph.first)

    def year_events(self) -> list[list[str]]:
        tgraph = self.tgraph.value
        return tgraph.events(self.current_year()) if tgraph is not None else []

    def year_note(self) -> str:
        tgraph = self.tgraph.value
        if self.world.value is None:
            return "Generate a world first."
        if tgraph is None:
            return (
                "This world does not change over time. Set Change over time to light or heavy "
                "and generate again to step through its years."
            )
        return tgraph.note

    def document_rows_now(self) -> list[dict]:
        """The document table: all of them, or those dated in or before the chosen year."""
        as_of = self.current_year() if self.viewing_time() else None
        return model.document_rows(self.documents.value, as_of)

    def highlight(self) -> dict | None:
        """What the graph emphasises: the clicked sentence's fact, else the picked question's path.
        Only what is drawn: a sampled graph may lack part of a path."""
        fact = model.fact_highlight(
            self.index.value, self.picked_fact.value, every_fact=self.viewing_time()
        )
        wanted = fact or model.highlight_for(self.index.value, self.picked_question())
        if wanted is None and self.viewing_time():
            return self.tgraph.value.changed(self.current_year())  # every id is drawn
        if wanted is None or not (self.graph_note.value or self.viewing_time()):
            return wanted
        nodes, edges = self.drawn()
        nodes_drawn, edges_drawn = {n["id"] for n in nodes}, {e["id"] for e in edges}
        return {
            "nodes": [n for n in wanted["nodes"] if n in nodes_drawn],
            "edges": [e for e in wanted["edges"] if e in edges_drawn],
        }

    def focus(self) -> str | None:
        """Where the graph centres: the clicked fact's subject, else the picked question's start,
        when the graph draws it."""
        index = self.index.value
        if index is not None and self.picked_fact.value in index.facts:
            target = index.facts[self.picked_fact.value].subject
        else:
            question = self.picked_question()
            target = question.start if question else None
        if target is None or not self.graph_note.value:
            return target
        return target if any(n["id"] == target for n in self.drawn()[0]) else None


def build_ui(store: workspaces.Store | None = None) -> UI:
    """Build one viewer's UI: independent signals, so two tabs never share a world. Workspaces
    are the one shared thing: they live on disk in ``store`` (``~/.kggen/studio`` by default)."""
    start = model.preset_values(model.DEFAULT_PRESET, model.DEFAULT_DIFFICULTY)
    preset = indah.Signal(model.DEFAULT_PRESET)
    difficulty = indah.Signal(model.DEFAULT_DIFFICULTY)
    evolution = indah.Signal(model.DEFAULT_EVOLUTION)
    persona = indah.Signal(model.DEFAULT_PERSONA)
    timeline = indah.Signal(model.DEFAULT_TIMELINE)
    phrasing = indah.Signal(model.DEFAULT_PHRASING)
    name_style = indah.Signal(model.DEFAULT_NAME_STYLE)
    values = {name: indah.Signal(value) for name, value in start.items()}
    status = indah.Signal("Check your choices, then generate.")

    def apply_preset() -> None:
        for name, value in model.preset_values(
            preset.value, difficulty.value, evolution.value
        ).items():
            values[name].set(value)

    indah.effect(apply_preset)  # also runs once now, which fills the form from the first preset

    question_values = {k: indah.Signal(v) for k, v in model.QUESTION_DEFAULTS.items()}

    def apply_persona() -> None:
        for name, value in model.persona_values(persona.value).items():
            question_values[name].set(value)

    indah.effect(apply_persona)  # choosing who asks fills the type and hop weights

    ui = UI(
        root=None,  # type: ignore[arg-type]
        preset=preset,
        difficulty=difficulty,
        evolution=evolution,
        persona=persona,
        timeline=timeline,
        phrasing=phrasing,
        name_style=name_style,
        values=values,
        status=status,
        entities=indah.Signal("-"),
        facts=indah.Signal("-"),
        world=indah.Signal(None),
        checks=indah.Signal([]),
        graph=indah.Signal(([], [])),
        graph_note=indah.Signal(""),
        tgraph=indah.Signal(None),
        time_on=indah.Signal(False),
        offset=indah.Signal(0),
        selected=indah.Signal(None),
        window=indah.Signal(300),
        index=indah.Signal(None),
        question_values=question_values,
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
        store=store if store is not None else workspaces.Store(),
        ws_rows=indah.Signal([]),
        ws_current=indah.Signal(None),
        ws_picked=indah.Signal(None),
        ws_name=indah.Signal(""),
        ws_note=indah.Signal(""),
        rev_rows=indah.Signal([]),
        rev_picked=indah.Signal(None),
        ws_confirm=indah.Signal(False),
        step=indah.Signal(0),
        tab=indah.Signal(0),
    )
    ui.refresh_workspaces()

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

    def workspace_picked() -> None:
        slug = ui.ws_picked.value
        if slug is not None:
            ui.open_workspace(slug)

    for effect in (question_picked, fact_picked, document_opened, workspace_picked):
        indah.effect(effect)

    def group_inputs(group: str) -> list[indah.Component]:
        return [field(k, values[k.name]) for k in model.KNOBS if k.group == group]

    def verdict() -> str:
        rows = ui.checks.value
        if not rows:
            return "Generate a world to compare what it realized against what you asked for."
        off = [r[0] for r in rows if r[3] != "ok"]
        return "All knobs are within tolerance." if not off else "Off target: " + ", ".join(off)

    generate_card = wizard(ui, values)
    setup = indah.Column(
        [
            generate_card,
            indah.Show(
                lambda: ui.world.value is not None,
                [
                    indah.Row(
                        [
                            indah.Stat(lambda: ui.entities.value, label="Entities"),
                            indah.Stat(lambda: ui.facts.value, label="Facts"),
                        ]
                    ),
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
                ],
            ),
        ]
    )
    tabs = indah.Tabs(
        [
            setup,
            indah.Column([world_card(ui), *entity_cards(ui)]),
            questions_card(ui),
            documents_card(ui),
            export_card(ui),
            history_card(ui),
        ],
        labels=["Setup", "World", "Questions", "Documents", "Export", "History"],
        active=ui.tab,
    )
    ui.root = indah.AppShell(
        rail=rail(ui),
        main=[indah.Text(ui.workspace_title), tabs],
        rail_width="280px",
    )
    return ui


def world_card(ui: UI) -> indah.Component:
    """The graph, or a note when this indah has no Graph yet."""
    if not HAS_GRAPH:
        return indah.Card(
            title="The world",
            children=[indah.Text("The graph view needs indah 0.12 or newer; yours has no Graph.")],
        )
    drawn = indah.computed(ui.drawn)
    graph = indah.Graph(
        lambda: drawn.value[0],
        lambda: drawn.value[1],
        selected=ui.selected,
        focus_size=ui.drawn_in_full,
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
            years_panel(ui),
        ],
    )


def years_panel(ui: UI) -> indah.Component:
    """Step the world through its years: a slider, buttons, and what each year changed."""
    return indah.Column(
        [
            indah.Switch(ui.time_on, label="Step through the years"),
            indah.Text(ui.year_note),
            indah.Show(
                lambda: ui.viewing_time(),
                [
                    indah.Text(lambda: f"Showing {ui.current_year()}"),
                    indah.Text(YEAR_LEGEND),
                    indah.Slider(
                        ui.offset,
                        min=0,
                        max=model.MAX_YEAR_STEPS,
                        step=1,
                        label="Years after the first year",
                        show_value=True,
                    ),
                    indah.Row(
                        [
                            indah.Button("Previous change", lambda: ui.step_year(-1, True)),
                            indah.Button("Previous year", lambda: ui.step_year(-1, False)),
                            indah.Button("Next year", lambda: ui.step_year(1, False)),
                            indah.Button("Next change", lambda: ui.step_year(1, True)),
                        ]
                    ),
                    indah.Table(
                        rows=indah.computed(
                            lambda: [{"sign": r[0], "what": r[1]} for r in ui.year_events()]
                        ),
                        columns=[
                            indah.Col("sign", ""),
                            indah.Col("what", "What changed", wrap=True),
                        ],
                        key="what",
                        height="220px",
                        label="What changed in this year",
                    ),
                ],
            ),
        ]
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
            indah.Select(
                ui.persona,
                options=list(model.PERSONA_CHOICES),
                label="Who asks (newcomer, learner, expert, mixed)",
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
        rows=indah.computed(ui.document_rows_now),
        columns=[
            indah.Col("id", "ID", kind="mono"),
            indah.Col("date", "Dated", kind="mono"),
            indah.Col("title", "About"),
            indah.Col("sentences", "Sentences", kind="number"),
            indah.Col("words", "Words", kind="number"),
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
            indah.Text(lambda: model.document_text(ui.picked_document())),
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


def choice_cards(signal: indah.Signal, choices: tuple) -> indah.Component:
    """One card per choice: its title, what it means, and a button; the chosen one says so. (A
    radio with rich options is an indah gap, GRA-193.)"""
    cards = []
    for value, title, blurb in choices:
        cards.append(
            indah.Card(
                title=title,
                children=[
                    indah.Text(blurb),
                    indah.Text(lambda value=value: "Chosen" if signal.value == value else ""),
                    indah.Button(
                        "Choose",
                        lambda value=value: signal.set(value),
                        variant="tonal",
                    ),
                ],
            )
        )
    return indah.Grid(cards)


def wizard(ui: UI, values: dict[str, indah.Signal]) -> indah.Component:
    """Making a dataset as five short steps instead of one long form. Every step edits the same
    signals the form always did, so a workspace saves and restores them unchanged."""

    def knobs(group: str) -> list[indah.Component]:
        return [field(k, values[k.name]) for k in model.KNOBS if k.group == group]

    start = indah.Show(
        lambda: ui.step.value == 0,
        [
            indah.Text("How big should the world be? A preset is a complete answer."),
            choice_cards(ui.preset, model.PRESET_CHOICES),
            indah.Button("Skip to review", lambda: ui.go(4), variant="tonal"),
        ],
    )
    world = indah.Show(
        lambda: ui.step.value == 1,
        [
            indah.Text("How hard should the text be to read?"),
            choice_cards(ui.difficulty, model.DIFFICULTY_CHOICES),
            indah.Text("What should the names look like?"),
            choice_cards(ui.name_style, model.NAME_CHOICES),
            indah.Expander(
                [
                    indah.Column([indah.Text(title), indah.Grid(knobs(group))])
                    for group, title in model.GROUPS.items()
                    if group != "churn"
                ],
                label="Fine-tune the numbers",
            ),
        ],
    )
    time = indah.Show(
        lambda: ui.step.value == 2,
        [
            indah.Text("Does the world change over the years?"),
            choice_cards(ui.evolution, model.EVOLUTION_CHOICES),
            indah.Text("How are the documents dated?"),
            choice_cards(ui.timeline, model.TIMELINE_CHOICES),
            indah.Expander([indah.Grid(knobs("churn"))], label="Fine-tune the changes"),
        ],
    )
    questions = indah.Show(
        lambda: ui.step.value == 3,
        [
            indah.Text("Who is asking the questions?"),
            choice_cards(ui.persona, model.PERSONA_CHOICES_TEXT),
            indah.Text("How should the questions and text be worded?"),
            choice_cards(ui.phrasing, model.WORDING_CHOICES),
        ],
    )
    review = indah.Show(
        lambda: ui.step.value == 4,
        [
            indah.Text("This is what you are asking for. Go back to change any of it."),
            indah.DataFrame(
                lambda: {
                    "columns": ["Your choices"],
                    "rows": [[line] for line in model.review_lines(ui.settings())],
                }
            ),
            indah.Button("Generate world", ui.generate_and_show),
            indah.Text(lambda: ui.status.value),
        ],
    )
    return indah.Card(
        title="Make a dataset",
        children=[
            indah.Steps(ui.step_items, orientation="horizontal"),
            indah.Text(lambda: model.estimate(ui.current_values())),
            start,
            world,
            time,
            questions,
            review,
            indah.Row(
                [
                    indah.Show(
                        lambda: ui.step.value > 0,
                        [indah.Button("Back", lambda: ui.go(ui.step.value - 1), variant="tonal")],
                    ),
                    indah.Show(
                        lambda: ui.step.value < len(model.WIZARD_STEPS) - 1,
                        [indah.Button("Next", lambda: ui.go(ui.step.value + 1))],
                    ),
                ]
            ),
        ],
    )


def rail(ui: UI) -> indah.Component:
    """The Studio rail: the saved workspaces, and a way to name a new one from the form."""
    table = indah.Table(
        rows=ui.ws_rows,
        columns=[indah.Col("name", "Workspace"), indah.Col("saved", "Saved", kind="mono")],
        key="slug",
        selected=ui.ws_picked,
        height="260px",
        label="Workspaces",
    )
    return indah.Column(
        [
            indah.Text("Studio"),
            table,
            indah.TextInput(ui.ws_name, label="Name", placeholder="Harbour towns, heavy churn"),
            indah.Button("New workspace from this form", ui.create_workspace),
            indah.Text(lambda: ui.ws_note.value),
        ]
    )


def history_card(ui: UI) -> indah.Component:
    """The open workspace's revisions, and what can be done to the workspace."""
    table = indah.Table(
        rows=ui.rev_rows,
        columns=[
            indah.Col("id", "Revision", kind="number"),
            indah.Col("saved", "Saved", kind="mono"),
            indah.Col("what", "Settings", wrap=True),
        ],
        key="id",
        selected=ui.rev_picked,
        height="260px",
        label="Revisions",
    )
    return indah.Card(
        title="History",
        children=[
            indah.Text(
                "Every Generate in a workspace that changed its settings keeps a revision. The "
                "data is not stored: it is made again from the settings, so a revision costs a few "
                "lines."
            ),
            table,
            indah.Button("Restore this revision", ui.restore_revision, variant="tonal"),
            indah.TextInput(ui.ws_name, label="Name for a rename or a copy"),
            indah.Row(
                [
                    indah.Button("Rename workspace", ui.rename_workspace, variant="tonal"),
                    indah.Button("Duplicate workspace", ui.duplicate_workspace, variant="tonal"),
                    indah.Button("Delete workspace", lambda: ui.ws_confirm.set(True)),
                ]
            ),
            indah.Modal(
                [
                    indah.Text(lambda: f"Delete {ui.workspace_title()}? Its history goes too."),
                    indah.Row(
                        [
                            indah.Button("Delete", ui.delete_workspace),
                            indah.Button("Keep it", lambda: ui.ws_confirm.set(False)),
                        ]
                    ),
                ],
                open=ui.ws_confirm,
                title="Delete workspace",
                on_close=lambda: ui.ws_confirm.set(False),
            ),
        ],
    )


def build_session() -> indah.Session:
    return indah.Session(build_ui().root)


def serve(port: int | None = None, host: str = "127.0.0.1") -> None:
    """Start the app; blocks until stopped."""
    indah.launch(
        indah.create_app(session_factory=build_session, title="kggen"), port=port, host=host
    )
