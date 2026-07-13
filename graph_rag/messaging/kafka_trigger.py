"""Kafka messaging seam for ingest triggers (N2/N3).

Two thin objects over ``kafka-python``:

* :class:`KafkaTriggerPublisher` implements the ``TriggerPublisher`` port —
  ``POST /ingest`` publishes an :class:`~graph_rag.models.IngestTrigger` through it.
* :class:`KafkaTriggerConsumer` is the **thin driver** (ADR-0001): it consumes
  trigger messages, deserialises each to an ``IngestTrigger`` and hands it to an
  **injected** ``handler``. It does *not* import the orchestrator — the composition
  root (main.py) wires ``process_document`` in as the handler. Per ADR-0001 the
  log-and-drop of a failing *document* is the orchestrator's job; the consumer only
  guards against a *malformed message* wedging the loop, then keeps delivering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from kafka import KafkaConsumer, KafkaProducer

from graph_rag.logging import get_logger
from graph_rag.models import IngestTrigger

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["KafkaTriggerPublisher", "KafkaTriggerConsumer"]

_logger = get_logger(__name__)

#: Type of the callback the consumer drives for each received trigger.
TriggerHandler = Callable[[IngestTrigger], None]


class KafkaTriggerPublisher:
    """Publish :class:`~graph_rag.models.IngestTrigger`s to the Kafka trigger topic.

    Implements the ``TriggerPublisher`` port. Serialises via ``trigger.to_json()``
    and produces UTF-8 encoded JSON to ``topic``.
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        """Build a producer against ``bootstrap_servers`` publishing to ``topic``.

        Args:
            bootstrap_servers: Kafka bootstrap servers, e.g. ``"kafka:9092"``.
            topic: The ingest-trigger topic to publish to.
        """
        self._topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: v.encode("utf-8"),
        )
        _logger.debug(
            "KafkaTriggerPublisher initialised bootstrap=%s topic=%s",
            bootstrap_servers,
            topic,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> KafkaTriggerPublisher:
        """Build a publisher from a :class:`~graph_rag.config.Settings`.

        Reads ``kafka_bootstrap_servers`` and ``ingest_trigger_topic``.
        """
        return cls(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            topic=settings.ingest_trigger_topic,
        )

    def publish(self, trigger: IngestTrigger) -> None:
        """Publish ``trigger`` to the ingest-trigger topic (JSON value)."""
        self._producer.send(self._topic, value=trigger.to_json())
        self._producer.flush()
        _logger.info(
            "published trigger bucket=%s object_key=%s topic=%s",
            trigger.bucket,
            trigger.object_key,
            self._topic,
        )

    def close(self) -> None:
        """Flush pending messages and close the underlying producer."""
        self._producer.flush()
        self._producer.close()


class KafkaTriggerConsumer:
    """Thin Kafka driver: consume triggers and hand each to an injected ``handler``.

    The ``handler`` is injected (ADR-0001/ADR-0010) so the consumer stays decoupled
    from the orchestrator and both are independently testable. A malformed message
    (undeserialisable, or a handler that raises) is logged and dropped rather than
    wedging the loop.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        handler: TriggerHandler,
        *,
        group_id: str = "graph-rag-ingest",
        auto_offset_reset: str = "earliest",
    ) -> None:
        """Build a consumer that drives ``handler`` for each received trigger.

        Args:
            bootstrap_servers: Kafka bootstrap servers, e.g. ``"kafka:9092"``.
            topic: The ingest-trigger topic to consume.
            handler: Callback invoked with each deserialised ``IngestTrigger``.
            group_id: Kafka consumer group id.
            auto_offset_reset: Where to start when the group has no committed
                offset (``"earliest"`` so a fresh consumer sees prior messages).
        """
        self._topic = topic
        self._handler = handler
        # Deserialise to bytes here and parse in the loop, so a malformed payload
        # surfaces as a caught exception per-message rather than breaking iteration.
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=None,
        )
        self._running = False
        _logger.debug(
            "KafkaTriggerConsumer initialised bootstrap=%s topic=%s group_id=%s",
            bootstrap_servers,
            topic,
            group_id,
        )

    def _dispatch(self, raw_value: bytes | None) -> bool:
        """Deserialise one raw message value and invoke the handler.

        Returns ``True`` if the message was delivered to the handler, ``False`` if
        it was logged-and-dropped (bad payload or handler error). Never raises.
        """
        try:
            trigger = IngestTrigger.from_json(raw_value)
        except Exception:  # noqa: BLE001 — a bad message must not wedge the loop.
            _logger.exception("dropping malformed trigger message: %r", raw_value)
            return False
        _logger.info("received trigger bucket=%s object_key=%s", trigger.bucket, trigger.object_key)
        try:
            self._handler(trigger)
        except Exception:  # noqa: BLE001 — handler failure must not wedge the loop.
            _logger.exception(
                "handler failed for trigger bucket=%s object_key=%s",
                trigger.bucket,
                trigger.object_key,
            )
            return False
        return True

    def poll_once(self, timeout_ms: int = 1000, max_records: int | None = None) -> int:
        """Poll a single batch, dispatch each message, and return the count handled.

        Deterministic entry point for tests/contract: it does one ``poll`` and
        returns rather than looping forever.

        Args:
            timeout_ms: How long to block waiting for records.
            max_records: Optional cap on records fetched in this poll.

        Returns:
            The number of messages successfully delivered to the handler.
        """
        batches = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        handled = 0
        for records in batches.values():
            for record in records:
                if self._dispatch(record.value):
                    handled += 1
        return handled

    def run_forever(self, poll_timeout_ms: int = 1000) -> None:
        """Consume and dispatch messages until :meth:`stop` is called.

        Args:
            poll_timeout_ms: Per-iteration poll timeout.
        """
        self._running = True
        _logger.info("consumer loop started topic=%s", self._topic)
        try:
            while self._running:
                self.poll_once(timeout_ms=poll_timeout_ms)
        finally:
            _logger.info("consumer loop stopped topic=%s", self._topic)

    # Backwards-friendly alias: the brief allows ``run`` or ``poll_once``.
    run = run_forever

    def stop(self) -> None:
        """Signal :meth:`run_forever` to exit after its current poll."""
        self._running = False

    def close(self) -> None:
        """Stop the loop and close the underlying consumer."""
        self.stop()
        self._consumer.close()
