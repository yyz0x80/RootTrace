"""Small Python 3.6-compatible JUnit adapter for Django's native test runner."""

import os
import unittest
from xml.etree import ElementTree

from django.test.runner import DiscoverRunner


class RootTraceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = []

    def _record(self, test, kind=None, detail=None):
        name = test.id() if hasattr(test, "id") else str(test)
        self.cases.append((name[:300], kind, (detail or "")[:2000]))

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test)

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self._record(test, "failure", self._exc_info_to_string(error, test))

    def addError(self, test, error):
        super().addError(test, error)
        self._record(test, "error", self._exc_info_to_string(error, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)

    def addExpectedFailure(self, test, error):
        super().addExpectedFailure(test, error)
        self._record(test, "skipped", "expected failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "failure", "unexpected success")


class RootTraceRunner(DiscoverRunner):
    def get_resultclass(self):
        return RootTraceResult

    def run_suite(self, suite, **kwargs):
        result = super().run_suite(suite, **kwargs)
        root = ElementTree.Element("testsuite", {
            "tests": str(len(result.cases)),
            "failures": str(sum(kind == "failure" for _, kind, _ in result.cases)),
            "errors": str(sum(kind == "error" for _, kind, _ in result.cases)),
            "skipped": str(sum(kind == "skipped" for _, kind, _ in result.cases)),
        })
        for name, kind, detail in result.cases:
            case = ElementTree.SubElement(root, "testcase", {"name": name})
            if kind is not None:
                ElementTree.SubElement(case, kind).text = detail
        path = os.environ["ROOTTRACE_JUNIT_PATH"]
        ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return result
