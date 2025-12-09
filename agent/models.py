from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

IntentLiteral = Literal["search", "summarize", "sentiment", "none"]
SearchModeLiteral = Literal["fast", "deep"]


class AgentState(BaseModel):
    """Состояние графа на каждом шаге."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_query: str
    detected_intents: List[IntentLiteral] = Field(default_factory=list)
    intermediate_result: Optional[str] = None
    final_answer: Optional[str] = None
    next_intent_index: int = 0  # служебный указатель прогресса
    search_mode: SearchModeLiteral = "fast"


class IntentPrediction(BaseModel):
    """Структурированный ответ для узла распознавания интентов."""

    intents: List[IntentLiteral] = Field(
        default_factory=list,
        description="Список интентов из набора search|summarize|sentiment|none",
    )


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResults(BaseModel):
    results: List[SearchResult]
