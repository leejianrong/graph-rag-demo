"""The kggen UI: pick a preset, set the knobs and a seed, generate a world, see its size."""

from __future__ import annotations

from dataclasses import dataclass

import indah

from kggen_ui import model


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

    def current_values(self) -> dict[str, float]:
        return {name: signal.value for name, signal in self.values.items()}

    def generate(self) -> None:
        try:
            world = model.generate_world(self.current_values())
        except ValueError as exc:
            self.status.set(str(exc))
            return
        summary = model.summarize(world)
        self.world.set(world)
        self.entities.set(f"{summary.entities:,}")
        self.facts.set(f"{summary.facts:,}")
        self.status.set(f"Generated with seed {summary.seed}.")


def build_ui() -> UI:
    """Build one viewer's UI: independent signals, so two tabs never share a world."""
    start = model.preset_values(model.DEFAULT_PRESET)
    preset = indah.Signal(model.DEFAULT_PRESET)
    values = {name: indah.Signal(value) for name, value in start.items()}
    status = indah.Signal("Choose a preset or set the knobs, then generate.")
    entities, facts = indah.Signal("-"), indah.Signal("-")
    world = indah.Signal(None)

    def apply_preset() -> None:
        for name, value in model.preset_values(preset.value).items():
            values[name].set(value)

    indah.effect(apply_preset)  # also runs once now, which fills the form from the first preset

    ui = UI(None, preset, values, status, entities, facts, world)  # type: ignore[arg-type]
    inputs = [
        indah.Number(values[k.name], min=k.min, max=k.max, step=k.step, label=k.label)
        for k in model.KNOBS
    ]
    ui.root = indah.Column(
        [
            indah.Card(
                title="Generate a world",
                children=[
                    indah.Select(preset, options=list(model.PRESET_NAMES), label="Preset"),
                    indah.Grid(inputs),
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
        ]
    )
    return ui


def build_session() -> indah.Session:
    return indah.Session(build_ui().root)


def serve(port: int | None = None, host: str = "127.0.0.1") -> None:
    """Start the app; blocks until stopped."""
    indah.launch(
        indah.create_app(session_factory=build_session, title="kggen"), port=port, host=host
    )
