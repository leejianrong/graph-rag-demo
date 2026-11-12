# ADR-0016: Ingest triggers go through a Valkey stream by default; Kafka stays selectable

- Status: Accepted (2026-11-12)
- Related: ADR-0001 (thin consumer driver), ADR-0010 (ports and fakes); GRA-31 (workers), GRA-138 (hosted stack), GRA-146

## Context

Kafka carried the ingest trigger: one small JSON message per document. For that job it is heavy (a JVM broker, no login
in our stack, a partition count that caps the workers). What Kafka gave us beyond a queue was an ordering rule: the
publisher keyed a trigger by `scope|object_key`, so one document's triggers stayed on one partition and one worker.
The orchestrator reads the stored object when it processes, so two workers on one document would interleave graph writes.

## Decision

`TRIGGER_BACKEND` is `redis` (the default) or `kafka`; one small factory (`graph_rag/messaging/factory.py`) builds the
publisher and consumer, so main.py and worker.py hold no backend branches. `redis` means Redis Streams over Valkey (BSD
licence, same protocol, one `redis` client). `INGEST_PARTITIONS` and the Kafka settings apply to `kafka` only.

**Delivery is at-least-once.** `XADD` publishes. A worker reads with `XREADGROUP` under a consumer name unique to its
process and one trigger at a time, and `XACK`s only after the handler returns (a handler that raises is acknowledged,
as with Kafka: failing a document is the orchestrator's job, ADR-0001). The acknowledged entry is then `XDEL`ed so the
stream stays small. Unacknowledged entries are never trimmed, and Valkey runs `--appendonly yes` with `noeviction`, so
queued triggers survive a restart and a full server refuses writes rather than dropping one.

**Crash recovery by reclaim.** Each poll first runs `XAUTOCLAIM` for entries idle longer than
`REDIS_RECLAIM_IDLE_SECONDS` (default 900, since a document can take minutes). While a handler runs, a heartbeat
re-claims its own entry (resetting the idle clock) and renews the lock, so a slow document is not taken from a live
worker; only a worker that stopped renewing loses its trigger.

**Per-document exclusion by lock.** A worker takes `SET <stream>:lock:<sha256(scope|object_key)> <consumer> NX PX
REDIS_LOCK_SECONDS` (default 300) before the handler and releases it after the ack (check-and-delete under `WATCH`, so
it never frees another worker's lock). The TTL only has to outlive a dead worker, so it is kept below the reclaim time:
by the time a reclaimed trigger is tried, the dead worker's lock has lapsed. If the lock is busy, the worker `XADD`s the
trigger back at the tail, then acknowledges its copy, then waits a second; a crash between the two steps duplicates the
trigger, it cannot lose it. Nothing is dropped and nothing is acknowledged unhandled.

**Shutdown.** `stop()` (SIGTERM) lets the document in hand finish; with one entry per read, unread triggers stay in the
stream. A clean exit removes the worker's consumer from the group when it holds nothing.

## Consequences

- Ordering is weaker than Kafka's. Re-queued triggers go to the tail, so triggers for one document may run in a
  different order than they were published. This is accepted: processing reads the object as it is when the trigger
  runs, so the last run always sees the latest content, and the document id is deterministic.
- A document whose handler kills the worker (an out-of-memory kill, say) is reclaimed and tried again after each idle
  period, indefinitely. There is no dead-letter stream yet.
- A document in flight past a lock loss (a worker paused longer than the TTL, with the lock then taken by another) can
  run twice at once. The heartbeat makes this need a stall of the whole process; it is the standard limit of a lock
  without fencing, and the pipeline's delete-then-write graph checkpoint keeps the end state correct.
- The hosted stack requires `VALKEY_PASSWORD`, and the service refuses to start with auth on and none set. Kafka's
  no-login note in the threat model now applies only to the `kafka` backend.
- Compose: Valkey is in the default set; Kafka is the `kafka` profile (`TRIGGER_BACKEND=kafka docker compose --profile
  kafka up`). `app` and `worker` depend on `kafka` with `required: false`, which compose skips when the profile is off.
  Valkey still starts in Kafka mode; it is small and idle.
- Not verified here: several worker processes against a real Valkey under load, and `docker compose up` with each
  profile. The unit tests use fakeredis for the two-consumer cases and the contract test uses a real Valkey for
  round trip, reclaim, exclusion and restart.
