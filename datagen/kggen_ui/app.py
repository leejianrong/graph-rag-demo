"""The kggen UI: pick a preset, set the knobs and a seed, generate a world, see its size."""

from __future__ import annotations

from dataclasses import dataclass

import indah

from kggen_ui import model

# The Graph component arrived in indah 0.12; an older indah still gets the form and the stats.
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
    index: indah.Signal  # a model.WorldIndex for the detail panel

    def current_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.values.items()}

    def generate(self) -> None:
        try:
            world = model.generate_world(self.current_values())
        except ValueError as exc:
            self.status.set(str(exc))
            return
        summary = model.summarize(world)
        note = model.graph_limits_message(world)
        shown = ([], []) if note else model.graph_data(world)

        def apply() -> None:
            # The old selection names an entity that may not exist in the new world, so it goes
            # first; one batch means the graph never sees the new data with the old selection.
            self.selected.set(None)
            self.world.set(world)
            self.index.set(model.index_world(world))
            self.graph.set(shown)
            self.graph_note.set(note or "")
            self.checks.set(model.check_rows(world))
            self.entities.set(f"{summary.entities:,}")
            self.facts.set(f"{summary.facts:,}")
            self.status.set(f"Generated with seed {summary.seed}.")

        indah.batch(apply)


def build_ui() -> UI:
    """Build one viewer's UI: independent signals, so two tabs never share a world."""
    start = model.preset_values(model.DEFAULT_PRESET)
    preset = indah.Signal(model.DEFAULT_PRESET)
    values = {name: indah.Signal(value) for name, value in start.items()}
    status = indah.Signal("Choose a preset or set the knobs, then generate.")
    entities, facts = indah.Signal("-"), indah.Signal("-")
    world = indah.Signal(None)
    checks = indah.Signal([])
    graph = indah.Signal(([], []))
    graph_note = indah.Signal("")
    selected = indah.Signal(None)
    window = indah.Signal(300)
    index = indah.Signal(None)

    def apply_preset() -> None:
        for name, value in model.preset_values(preset.value).items():
            values[name].set(value)

    indah.effect(apply_preset)  # also runs once now, which fills the form from the first preset

    ui = UI(
        None,
        preset,
        values,
        status,
        entities,
        facts,
        world,
        checks,
        graph,
        graph_note,
        selected,
        window,
        index,
    )  # type: ignore[arg-type]

    def group_inputs(group: str) -> list[indah.Component]:
        return [
            indah.Number(values[k.name], min=k.min, max=k.max, step=k.step, label=k.label)
            for k in model.KNOBS
            if k.group == group
        ]

    def verdict() -> str:
        rows = checks.value
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
                    indah.Text(lambda: status.value),
                ],
            ),
            indah.Row(
                [
                    indah.Stat(lambda: entities.value, label="Entities"),
                    indah.Stat(lambda: facts.value, label="Facts"),
                ]
            ),
            *graph_cards(ui),
            indah.Card(
                title="Asked for vs. realized",
                children=[
                    indah.Text(verdict),
                    indah.DataFrame(
                        lambda: {
                            "columns": ["Knob", "Asked for", "Realized", "Result"],
                            "rows": checks.value,
                        }
                    ),
                ],
            ),
        ]
    )
    return ui


def graph_cards(ui: UI) -> list[indah.Component]:
    """The graph and the detail of the clicked entity, or a note when indah has no Graph yet."""
    if not HAS_GRAPH:
        return [
            indah.Card(
                title="Graph",
                children=[
                    indah.Text("The graph view needs indah 0.12 or newer; yours has no Graph.")
                ],
            )
        ]
    graph = indah.Graph(
        lambda: ui.graph.value[0],
        lambda: ui.graph.value[1],
        selected=ui.selected,
        focus_size=ui.window,
        height=520,
        max_nodes=model.MAX_GRAPH_NODES,
        label="The generated world: entities coloured by type",
    )
    facts = indah.DataFrame(
        lambda: {
            "columns": model.FACT_COLUMNS,
            "rows": model.fact_rows(ui.index.value, ui.selected.value),
        }
    )
    return [
        indah.Card(
            title="The world",
            children=[
                indah.Text(lambda: ui.graph_note.value or ""),
                graph,
                indah.Slider(ui.window, min=20, max=2000, step=10, label="Nodes drawn in full"),
            ],
        ),
        indah.Card(
            title="Selected entity",
            children=[
                indah.Text(lambda: model.entity_summary(ui.index.value, ui.selected.value)),
                facts,
            ],
        ),
    ]


def build_session() -> indah.Session:
    return indah.Session(build_ui().root)


def serve(port: int | None = None, host: str = "127.0.0.1") -> None:
    """Start the app; blocks until stopped."""
    indah.launch(
        indah.create_app(session_factory=build_session, title="kggen"), port=port, host=host
    )
