# Correct median for even-length inputs

`benchmark.statistics.median` returns the lower middle value for an even number
of observations. It should return the arithmetic mean of the two middle values.

Acceptance requirements:

- For an even-length input, return the mean of the two middle values as a float.
- Preserve the current result for odd-length inputs.
- Do not mutate the caller's list.
- Continue to raise `ValueError` for an empty list.
- Keep tests read-only and change only `benchmark/statistics.py`.

