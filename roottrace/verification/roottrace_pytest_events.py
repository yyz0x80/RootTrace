"""Stream pytest test outcomes without relying on a container JUnit file."""

from roottrace_event_protocol import emit

_reported = set()


def pytest_runtest_logreport(report):
    if report.when == "call":
        if hasattr(report, "wasxfail"):
            status = "skipped" if report.skipped else "failure"
        elif report.passed:
            status = "passed"
        elif report.skipped:
            status = "skipped"
        else:
            status = "failure"
    elif report.skipped:
        status = "skipped"
    elif report.failed:
        status = "error"
    else:
        return
    name = report.nodeid[:300]
    _reported.add(name)
    emit("case", name=name, status=status)


def pytest_collectreport(report):
    if report.failed:
        name = report.nodeid[:300]
        _reported.add(name)
        emit("case", name=name, status="error")


def pytest_sessionfinish(session, exitstatus):
    emit("end", exit_code=int(exitstatus), collected=int(session.testscollected),
         reported=len(_reported))
