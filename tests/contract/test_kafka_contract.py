"""Contract test: a published trigger is received and deserialised intact.

Proves the real :class:`KafkaTriggerPublisher` and :class:`KafkaTriggerConsumer`
round-trip an :class:`~graph_rag.models.IngestTrigger` through a real Kafka
(testcontainers): what the consumer hands the injected handler equals what was
published (ADR-0001). Skips cleanly when Docker is unavailable.

Marked ``contract`` — excluded from the fast ($0) suite.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.contract

# Import guard: testcontainers/kafka deps missing must skip, not error at collection.
try:
    from testcontainers.kafka import KafkaContainer

    from graph_rag.messaging.kafka_trigger import (
        KafkaTriggerConsumer,
        KafkaTriggerPublisher,
    )
    from graph_rag.models import IngestTrigger
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"Kafka contract deps unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def bootstrap_servers() -> Iterator[str]:
    """Bootstrap servers for a throwaway Kafka container (KRaft mode)."""
    try:
        container = KafkaContainer().with_kraft()
        container.start()
    except Exception as exc:  # noqa: BLE001 — Docker not available on this host.
        pytest.skip(f"Docker unavailable for Kafka contract test: {exc}")

    try:
        yield container.get_bootstrap_server()
    finally:
        container.stop()


def test_published_trigger_is_received_and_equal(bootstrap_servers: str) -> None:
    """A published IngestTrigger reaches the consumer's handler unchanged."""
    topic = f"ingest-triggers-{uuid.uuid4().hex[:8]}"
    received: list[IngestTrigger] = []

    consumer = KafkaTriggerConsumer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        handler=received.append,
        group_id=f"contract-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
    )
    publisher = KafkaTriggerPublisher(bootstrap_servers=bootstrap_servers, topic=topic)

    try:
        sent = IngestTrigger(bucket="documents", object_key="contract.md")
        publisher.publish(sent)

        # Poll a few times: give the broker/consumer group time to deliver.
        deadline_polls = 20
        while deadline_polls and not received:
            consumer.poll_once(timeout_ms=1000)
            deadline_polls -= 1

        assert received, "no trigger received within the poll window"
        assert received[0] == sent
        assert received[0].bucket == "documents"
        assert received[0].object_key == "contract.md"
    finally:
        publisher.close()
        consumer.close()
