"""
Root pytest configuration for SentinelSentry.
Resolves Python 3.14 on Windows terminal teardown deadlock.
"""

import os
import sys

total_collected = 0
tests_completed = 0
failures = 0


def pytest_collection_finish(session):
    global total_collected
    total_collected = len(session.items)


def pytest_runtest_logreport(report):
    global tests_completed, total_collected, failures
    if report.when == "call":
        tests_completed += 1
        if report.failed:
            failures += 1
        if total_collected > 0 and tests_completed >= total_collected:
            status_text = f"{tests_completed - failures} passed"
            if failures > 0:
                status_text += f", {failures} failed"
            sys.stdout.write(f"\n======================= {status_text} =======================\n")
            sys.stdout.flush()
            os._exit(1 if failures > 0 else 0)
