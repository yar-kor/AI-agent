from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .models import IntentPrediction, SearchResults
from .prompts import (
    FALLBACK_PROMPT,
    INTENT_PROMPT,
    SEARCH_PROMPT,
    SENTIMENT_PROMPT,
    SUMMARY_PROMPT,
)


def _parse_intents(output: Any) -> IntentPrediction:
    """Парсим строковый/контентный вывод в список интентов."""
    text = getattr(output, "content", None) or str(output)
    intents: list[str] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            intents = [str(item) for item in data]
    except Exception:
        pass
    return IntentPrediction(intents=intents)


def build_intent_chain(llm: ChatOpenAI):
    """Цепочка для детекции интентов с ручным парсером JSON-массива."""
    return INTENT_PROMPT | llm | RunnableLambda(_parse_intents)


def build_search_chain(llm: ChatOpenAI):
    """Цепочка для генерации фейковых результатов поиска."""
    return SEARCH_PROMPT | llm.with_structured_output(SearchResults)


def build_summary_chain(llm: ChatOpenAI):
    """Цепочка для суммаризации."""
    return SUMMARY_PROMPT | llm


def build_sentiment_chain(llm: ChatOpenAI):
    """Цепочка для определения тональности."""
    return SENTIMENT_PROMPT | llm


def build_fallback_chain(llm: ChatOpenAI):
    """Цепочка fallback-ответа."""
    return FALLBACK_PROMPT | llm
