"""Build the trigger publisher and consumer for the configured ``TRIGGER_BACKEND`` (GRA-146).

The one place that knows there is more than one backend; main.py and worker.py call these.
Each backend is imported on use, so an install that runs one never loads the other's client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from graph_rag.ports import TriggerPublisher

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.messaging.redis_trigger import TriggerHandler

__all__ = ["TriggerConsumer", "build_publisher", "build_consumer"]


class TriggerConsumer(Protocol):
    """What the service and the worker need from a trigger consumer, whichever backend."""

    def run_forever(self, poll_timeout_ms: int = ..., max_records: int | None = ...) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


def build_publisher(settings: Settings) -> TriggerPublisher:
    """The publisher ``POST /ingest`` hands triggers to."""
    if settings.trigger_backend == "kafka":
        from graph_rag.messaging.kafka_trigger import KafkaTriggerPublisher

        return KafkaTriggerPublisher.from_settings(settings)
    from graph_rag.messaging.redis_trigger import RedisTriggerPublisher

    return RedisTriggerPublisher.from_settings(settings)


def build_consumer(settings: Settings, handler: TriggerHandler) -> TriggerConsumer:
    """A consumer driving ``handler`` for each trigger: ``run_forever`` / ``stop`` / ``close``."""
    if settings.trigger_backend == "kafka":
        from graph_rag.messaging.kafka_trigger import KafkaTriggerConsumer

        return KafkaTriggerConsumer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            topic=settings.ingest_trigger_topic,
            handler=handler,
        )
    from graph_rag.messaging.redis_trigger import RedisTriggerConsumer

    return RedisTriggerConsumer.from_settings(settings, handler)
