"""Async LLM client. Two configured clients: `student` (small/fast) and `generator` (strong).

OpenAI-compatible, so the same code hits OpenAI, Qwen/DashScope, GLM/Zhipu, or a local
vLLM server just by swapping base_url + model in .env.
"""
from openai import AsyncOpenAI

from .config import settings

student_client = AsyncOpenAI(
    base_url=settings.student_base_url,
    api_key=settings.student_api_key or "not-set",
)

generator_client = AsyncOpenAI(
    base_url=settings.generator_base_url,
    api_key=settings.generator_api_key or "not-set",
)


async def student_chat(system: str, user: str, temperature: float = 0.7) -> str:
    """One student-tier completion."""
    resp = await student_client.chat.completions.create(
        model=settings.student_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


async def generator_chat(system: str, user: str, temperature: float = 0.4) -> str:
    """One generator-tier completion."""
    resp = await generator_client.chat.completions.create(
        model=settings.generator_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


async def verifier_chat(system: str, user: str, temperature: float = 0.0) -> str:
    """One grader completion. Uses `verifier_model` if set, else the generator model. Shares the
    generator client (same endpoint/key)."""
    resp = await generator_client.chat.completions.create(
        model=settings.verifier_model or settings.generator_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
