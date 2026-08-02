"""A deliberately-failing mutation, used to prove restoration is unconditional.

NOT collected by a normal test run — the filename does not match `test_*.py`.
It is executed in a SUBPROCESS by
`test_isolation_redteam.py::test_restoration_survives_a_failing_assertion`,
which asserts that the database came back healthy anyway.

This file is written the WRONG way on purpose: it disables RLS and then fails,
with no try/finally. That is the exact mistake the autouse repair fixture in
conftest.py exists to absorb, and the only way to prove the fixture works is to
make the mistake and observe the recovery. Do not "fix" this file.
"""

from sqlalchemy import text

from review_agent.data.db import get_owner_engine


def test_fails_midway_through_a_mutation(provisioned_db):
    with get_owner_engine().begin() as conn:
        conn.execute(text("ALTER TABLE artifacts DISABLE ROW LEVEL SECURITY"))

    # No try/finally, no restore. The suite would be compromised from here on.
    raise AssertionError("simulated mid-mutation failure")
