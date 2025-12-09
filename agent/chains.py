from __future__ import annotations

from langchain_openai import ChatOpenAI

from .models import IntentPrediction, SearchResults
from .prompts import (
    FALLBACK_PROMPT,
    INTENT_PROMPT,
    SEARCH_PROMPT,
    SENTIMENT_PROMPT,
    SUMMARY_PROMPT,
)


def build_intent_chain(llm: ChatOpenAI):
    """Цепочка для детекции интентов с типизированным выводом."""
    return INTENT_PROMPT | llm.with_structured_output(IntentPrediction)


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
