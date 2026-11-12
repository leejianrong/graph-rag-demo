"""Messaging seam: the ingest trigger queue and its thin consumer driver.

The ``TriggerPublisher`` port lives in ``graph_rag.ports`` and its in-memory fake in
``graph_rag.fakes``. ``redis_trigger`` (the default, Valkey) and ``kafka_trigger`` are the real
backends; ``factory`` picks one from ``TRIGGER_BACKEND`` (ADR-0016).
"""
