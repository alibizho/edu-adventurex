"""LangChain chat models over the same OpenAI-compatible endpoint as `app/llm.py` (DeepSeek by
default). Used by the curriculum builder: `.with_structured_output(...)` for scope/structure and
plain `.ainvoke(...)` for the Markdown teacher's notes. The raw-OpenAI wrappers in `app/llm.py`
stay for the untouched confusion/targeted/measure pipeline.
"""
from langchain_openai import ChatOpenAI

from .config import settings

_NON_THINKING = {"thinking": {"type": "disabled"}}


def generator_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Stronger tier — plan design + notes. Same base_url/model/key as `generator_chat`."""
    return ChatOpenAI(
        base_url=settings.generator_base_url,
        api_key=settings.generator_api_key or "not-set",
        model=settings.generator_model,
        temperature=temperature,
        extra_body=_NON_THINKING,
    )


def student_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Small/fast tier. Same base_url/model/key as `student_chat`."""
    return ChatOpenAI(
        base_url=settings.student_base_url,
        api_key=settings.student_api_key or "not-set",
        model=settings.student_model,
        temperature=temperature,
        extra_body=_NON_THINKING,
    )
