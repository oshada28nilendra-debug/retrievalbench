API clients may send sixty requests per minute. A 429 response includes a Retry-After header with the wait time in seconds.

Webhooks are retried after one minute, then five minutes, then thirty minutes. Use the event ID as an idempotency key.

API tokens can be revoked in developer settings. Revocation takes effect immediately; rotate credentials without committing secrets to source control.

Pagination uses a cursor returned as next_cursor. Continue until next_cursor is null. Each page holds at most one hundred items.
