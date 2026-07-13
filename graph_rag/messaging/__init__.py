"""Messaging seam: the Kafka trigger topic + thin consumer driver.

The ``TriggerPublisher`` port lives in ``graph_rag.ports`` and its in-memory fake
in ``graph_rag.fakes``. Real Kafka producer/consumer modules are only *added* to
this package by later work.
"""
