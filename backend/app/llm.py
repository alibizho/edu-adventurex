from openai import AsyncOpenAI

from .config import settings

_NON_THINKING = {"thinking": {"type": "disabled"}}

student_client = AsyncOpenAI(
    base_url=settings.student_base_url,
    api_key=settings.student_api_key or "not-set",
)

generator_client = AsyncOpenAI(
    base_url=settings.generator_base_url,
    api_key=settings.generator_api_key or "not-set",
)

async def student_chat(system: str, user: str, temperature: float = 0.7) -> str:
    resp = await student_client.chat.completions.create(
        model=settings.student_model,
        temperature=temperature,
        extra_body=_NON_THINKING,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""

async def generator_chat(system: str, user: str, temperature: float = 0.4) -> str:
    resp = await generator_client.chat.completions.create(
        model=settings.generator_model,
        temperature=temperature,
        extra_body=_NON_THINKING,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""

async def verifier_chat(system: str, user: str, temperature: float = 0.0) -> str:
    resp = await generator_client.chat.completions.create(
        model=settings.verifier_model or settings.generator_model,
        temperature=temperature,
        extra_body=_NON_THINKING,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
