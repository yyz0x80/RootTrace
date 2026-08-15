# Parse quoted CSV fields correctly

`csv_demo.parse_record` splits on every comma, including commas inside quoted
fields. Replace the ad-hoc parsing with standards-compliant single-record CSV
parsing.

Acceptance requirements:

- A comma inside a quoted field must remain part of that field.
- Escaped double quotes inside quoted fields must be decoded correctly.
- Preserve leading and trailing spaces inside quoted fields.
- Preserve the current result for simple unquoted records, including trimming
  incidental spaces around unquoted fields.
- Use only the Python standard library.
- Keep tests read-only and change only `csv_demo/__init__.py`.

