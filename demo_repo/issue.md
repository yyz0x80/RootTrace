# Reject invalid pagination parameters

The paginate function should reject invalid pagination arguments.

Acceptance requirements:

- Raise ValueError when page is less than 1.
- Raise ValueError when page_size is less than 1.
- Preserve the existing behavior for valid page and page_size values.