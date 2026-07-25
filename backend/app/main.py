from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from .api.material_routes import router as material_router
from .api.plan_routes import router as plan_router
from .api.routes import router
from .config import settings
from .store import store

@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.init()
    yield
    await store.dispose()

app = FastAPI(title="wut", version="0.1.0", lifespan=lifespan)

allowed_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(OpenAIError)
async def handle_llm_error(_request: Request, error: OpenAIError) -> JSONResponse:
    if isinstance(error, AuthenticationError):
        return JSONResponse(
            status_code=401,
            content={
                "detail": (
                    "THE LANGUAGE MODEL API KEY IS INVALID. "
                    "UPDATE backend/.env AND RESTART THE BACKEND."
                )
            },
        )
    if isinstance(error, RateLimitError):
        return JSONResponse(
            status_code=429,
            content={
                "detail": "LANGUAGE MODEL RATE LIMIT OR BALANCE ERROR. CHECK YOUR LLM ACCOUNT."
            },
        )
    if isinstance(error, APITimeoutError):
        return JSONResponse(
            status_code=504,
            content={"detail": "THE LANGUAGE MODEL REQUEST TIMED OUT. PLEASE TRY AGAIN."},
        )
    if isinstance(error, APIConnectionError):
        return JSONResponse(
            status_code=502,
            content={"detail": "THE BACKEND CANNOT REACH THE LANGUAGE MODEL API."},
        )
    if isinstance(error, APIStatusError):
        return JSONResponse(
            status_code=error.status_code if 400 <= error.status_code < 600 else 502,
            content={
                "detail": f"THE LANGUAGE MODEL REJECTED THE REQUEST ({error.status_code})."
            },
        )
    return JSONResponse(
        status_code=502,
        content={"detail": "THE LANGUAGE MODEL REQUEST FAILED."},
    )

app.include_router(router)
app.include_router(plan_router)
app.include_router(material_router)

@app.get("/")
async def root() -> dict:
    return {"service": "wut", "docs": "/docs"}
