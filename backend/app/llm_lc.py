from langchain_openai import ChatOpenAI

from .config import settings

_NON_THINKING = {"thinking": {"type": "disabled"}}

def generator_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.generator_base_url,
        api_key=settings.generator_api_key or "not-set",
        model=settings.generator_model,
        temperature=temperature,
        extra_body=_NON_THINKING,
    )
