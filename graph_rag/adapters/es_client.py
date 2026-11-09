"""The Elasticsearch client both stores use, signed in when the settings carry a password."""

from __future__ import annotations

from elasticsearch import Elasticsearch

from graph_rag.config import Settings


def es_client(settings: Settings) -> Elasticsearch:
    """A client for ``settings.elasticsearch_url``; basic auth only when a password is set."""
    if settings.elasticsearch_password:
        return Elasticsearch(
            hosts=[settings.elasticsearch_url],
            basic_auth=(settings.elasticsearch_user, settings.elasticsearch_password),
        )
    return Elasticsearch(hosts=[settings.elasticsearch_url])
