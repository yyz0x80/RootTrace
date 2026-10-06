"""Stream Django's native test results to RootTrace (Python 3.6+)."""

import unittest

from django.test.runner import DiscoverRunner
from roottrace_event_protocol import emit


class RootTraceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reported = set()

    def _record(self, test, status):
        name = test.id() if hasattr(test, "id") else str(test)
        name = name[:300]
        self.reported.add(name)
        emit("case", name=name, status=status)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self._record(test, "failure")

    def addError(self, test, error):
        super().addError(test, error)
        self._record(test, "error")

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error is not None:
            status = "failure" if issubclass(error[0], test.failureException) else "error"
            self._record(test, status)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped")

    def addExpectedFailure(self, test, error):
        super().addExpectedFailure(test, error)
        self._record(test, "skipped")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "failure")


class RootTraceRunner(DiscoverRunner):
    def get_resultclass(self):
        return RootTraceResult

    def run_suite(self, suite, **kwargs):
        result = super().run_suite(suite, **kwargs)
        emit("end", exit_code=int(not result.wasSuccessful()),
             reported=len(result.reported))
        return result
