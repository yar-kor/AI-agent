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
    """Парсим строковый/контентный вывод в список интентов"""
    text = getattr(output, "content", None) or str(output)
    intents: list[str] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            intents = [str(item) for item in data]
    except Exception:
        pass
    allowed = {"search", "summarize", "sentiment", "none"}
    filtered = []
    seen = set()
    for intent in intents:
        if intent in allowed and intent not in seen:
            filtered.append(intent)
            seen.add(intent)
    if not filtered:
        filtered = ["none"]
    return IntentPrediction(intents=filtered)


def build_intent_chain(llm: ChatOpenAI):
  
    return INTENT_PROMPT | llm | RunnableLambda(_parse_intents)


def build_search_chain(llm: ChatOpenAI):
 
    return SEARCH_PROMPT | llm.with_structured_output(SearchResults)


def build_summary_chain(llm: ChatOpenAI):

    return SUMMARY_PROMPT | llm


def build_sentiment_chain(llm: ChatOpenAI):
    
    return SENTIMENT_PROMPT | llm


def build_fallback_chain(llm: ChatOpenAI):

    return FALLBACK_PROMPT | llm
