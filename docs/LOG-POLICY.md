# Log policy

Logs are treated as holding user data, so they are kept free of it. Set in GRA-141; enforced by
`tests/unit/test_log_policy.py`.

## What may be logged (info and above)

- Ids: `document_id` (a hash), job ids, tenant names.
- Counts and sizes: nodes, edges, sentences, bytes, characters.
- Events: started, stopped, ingested, deleted, dropped, and which stage failed.

## What may not be logged (info and above)

- Question text, the answer, synthesised prose, or supporting sentences.
- Document text, source names (`source_id`, `source_url`, filenames) and object keys.
- API keys, store credentials and the LLM provider key.

A document is identified in logs by its `document_id`. To find which source it was, look it up in the
document store; the log does not say.

## Where this is not enough

- `logger.exception` prints a traceback, and an exception message from a library can quote its input.
  The orchestrator's own failure reasons name only the stage and the error type, but the traceback in
  the log is not scrubbed.
- Debug level is for development and may name store records by id. Do not run hosted at debug.
- The web server's access log records request paths, which include a document id on `/jobs` and
  `/documents`. Bodies and query strings are not logged by this code.
- Logs shipped to a collector are only as private as the collector.
