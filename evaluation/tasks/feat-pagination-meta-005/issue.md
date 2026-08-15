# Add pagination metadata

Callers need enough metadata to render page navigation without recomputing it.

Acceptance requirements:

- Preserve the existing `items` and `page` response fields.
- Add `page_size`, `total_items`, and `total_pages` fields.
- Compute `total_pages` with ceiling division; an empty collection has zero pages.
- A page beyond the end returns an empty `items` list with otherwise correct
  metadata.
- Continue to reject `page < 1` or `page_size < 1` with `ValueError`.
- Keep tests read-only and change only `benchmark/pagination.py`.

