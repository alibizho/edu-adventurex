"""Direct DbStore round-trip for the learning-plan tables (growth_paths, path_memory) against a
live Postgres — the durable counterpart to the in-memory checks in test_curriculum.py. Mirrors
test_db_roundtrip.py; needs DATABASE_URL + a reachable Postgres.

    DATABASE_URL=postgresql+asyncpg://ts:ts@localhost:5432/teachable PYTHONPATH=. \
        python tests/test_db_plan_roundtrip.py

It creates a throwaway path, verifies save/get + save_class + memory upsert + durability across a
fresh engine (a "restart"), then deletes its rows so the DB is left as it was found.
"""
import asyncio
import os
import sys
import uuid

from sqlalchemy import text

from app.schemas import ClassUnit, GrowthPath
from app.store.db import DbStore


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        raise AssertionError(msg)


async def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set; run against a reachable Postgres.")
        sys.exit(2)
    print(f"Connecting to {url.rsplit('@', 1)[-1]}")

    st = DbStore(url)
    pid = f"gp-test-{uuid.uuid4().hex[:8]}"
    try:
        await st.init()
        check(True, "init() created/verified plan tables")

        # -- save + get the whole GrowthPath (JSONB blob) --
        path = GrowthPath(
            path_id=pid,
            original_input="I want to learn physics",
            confirmed_topic="Classical Mechanics",
            total_classes=2,
            recommended_order=["c1", "c2"],
            classes=[
                ClassUnit(class_id="c1", title="Forces", objective="Define force."),
                ClassUnit(class_id="c2", title="Energy", objective="Define energy.", prerequisites=["c1"]),
            ],
        )
        await st.save_path(path)
        got = await st.get_path(pid)
        check(got is not None and got.confirmed_topic == "Classical Mechanics", "path save/get round-trip")
        check([c.class_id for c in got.classes] == ["c1", "c2"], "classes preserved and ordered")
        check(got.classes[1].prerequisites == ["c1"], "nested class fields round-trip")
        check(await st.get_path("gp-does-not-exist") is None, "unknown path returns None")

        # -- save_class updates ONE class's notes, in place --
        c1 = got.classes[0]
        c1.teacher_notes = "# Forces\nA push or a pull."
        c1.notes_generated = True
        await st.save_class(pid, c1)

        # -- durability across a RESTART: a brand-new engine reads the persisted blob --
        st2 = DbStore(url)
        try:
            re = await st2.get_path(pid)
            check(
                re.classes[0].notes_generated and re.classes[0].teacher_notes.startswith("# Forces"),
                "notes persist across a fresh engine (simulated restart)",
            )
            check(re.classes[1].notes_generated is False, "sibling class left untouched")
        finally:
            await st2.dispose()

        # -- cross-class memory: default, update, then upsert --
        mem = await st.get_memory(pid)
        check(mem.path_id == pid and mem.covered_concepts == [], "default memory for a new path")

        mem.covered_concepts.append("Forces")
        mem.asked_questions.append("What is a force?")
        mem.struggled.append("What is a force?")
        await st.update_memory(pid, mem)
        got_mem = await st.get_memory(pid)
        check(
            got_mem.covered_concepts == ["Forces"] and got_mem.asked_questions == ["What is a force?"],
            "memory update round-trip",
        )

        got_mem.understood.append("What is a force?")
        got_mem.struggled = []
        await st.update_memory(pid, got_mem)
        final = await st.get_memory(pid)
        check(final.understood == ["What is a force?"] and final.struggled == [], "memory upserts on path_id")

        print("\nALL PLAN DB ROUND-TRIP CHECKS PASSED")
    finally:
        async with st._engine.begin() as conn:
            await conn.execute(text("DELETE FROM growth_paths WHERE path_id = :p"), {"p": pid})
            await conn.execute(text("DELETE FROM path_memory WHERE path_id = :p"), {"p": pid})
        await st.dispose()
        print(f"cleaned up path {pid}")


if __name__ == "__main__":
    asyncio.run(main())
