"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.plan_routes import router as plan_router
from .api.routes import router
from .store import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init(): the DB store creates tables (so a fresh Postgres is usable with no migration step);
    # the memory store is a no-op. dispose() drops the engine connection pool on shutdown.
    await store.init()
    yield
    await store.dispose()


app = FastAPI(title="wut", version="0.1.0", lifespan=lifespan)

# Open CORS for the hackathon so the frontend (whatever port) can hit the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(plan_router)


@app.get("/")
async def root() -> dict:
    return {"service": "wut", "docs": "/docs"}
