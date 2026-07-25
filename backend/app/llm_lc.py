"""LangChain chat model over the same OpenAI-compatible endpoint as `app/llm.py` (DeepSeek by
default). Used by the curriculum builder: `.with_structured_output(...)` for scope/structure and
plain `.ainvoke(...)` for the Markdown teacher's notes. Everything else — the student, the
confusion/targeted/measure pipeline, the objective judge — goes through the raw-OpenAI wrappers in
`app/llm.py`, which is why only the generator tier is wrapped here.
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
