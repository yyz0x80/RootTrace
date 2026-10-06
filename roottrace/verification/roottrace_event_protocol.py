"""Emit bounded test events over the container's captured stdout (Python 3.6+)."""

import json
import os
import sys

PREFIX = "ROOTTRACE_TEST_EVENT:"
_sequence = 0


def emit(kind, **fields):
    global _sequence
    _sequence += 1
    token = os.environ["ROOTTRACE_EVENT_TOKEN"]
    record = {"kind": kind}
    record.update(fields)
    record["seq"] = _sequence
    sys.__stdout__.write(PREFIX + token + ":" + json.dumps(record, ensure_ascii=True) + "\n")
    sys.__stdout__.flush()
