"""Context store. `store` is the process-wide singleton the routes use — either in-memory dicts
(dev default, STORE_BACKEND=memory) or Postgres (STORE_BACKEND=db + DATABASE_URL). Both satisfy
the Store protocol in base.py.

The DB impl is imported lazily so a memory-only setup doesn't require sqlalchemy/asyncpg to be
installed.
"""
from ..config import settings


def get_store():
    if settings.store_backend == "db" and settings.database_url:
        from .db import DbStore

        return DbStore(settings.database_url)
    from .memory import MemoryStore

    return MemoryStore()


store = get_store()
