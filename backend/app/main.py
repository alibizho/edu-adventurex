"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(title="Teachable Student", version="0.1.0")

# Open CORS for the hackathon so the frontend (whatever port) can hit the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root() -> dict:
    return {"service": "teachable-student", "docs": "/docs"}
