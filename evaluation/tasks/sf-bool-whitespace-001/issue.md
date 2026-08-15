# Accept surrounding whitespace in boolean values

`benchmark.booleans.parse_bool` rejects otherwise valid values when they have
leading or trailing whitespace.

Acceptance requirements:

- Ignore leading and trailing Unicode whitespace before matching a value.
- Continue to match `true`, `yes`, `1`, `false`, `no`, and `0` without regard
  to letter case.
- Continue to raise `ValueError` for unsupported values and for a whitespace-only
  value.
- Preserve the `parse_bool(value: str) -> bool` interface.
- Keep tests read-only and change only `benchmark/booleans.py`.

