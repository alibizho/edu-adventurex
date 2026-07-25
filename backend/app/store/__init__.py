from ..config import settings

def get_store():
    if settings.store_backend == "db" and settings.database_url:
        from .db import DbStore

        return DbStore(settings.database_url)
    from .memory import MemoryStore

    return MemoryStore()

store = get_store()
