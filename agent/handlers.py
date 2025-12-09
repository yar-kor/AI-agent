from __future__ import annotations

import logging
import re
from typing import Dict
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from langchain_core.runnables import Runnable

from .chains import (
    build_fallback_chain,
    build_intent_chain,
    build_search_chain,
    build_sentiment_chain,
    build_summary_chain,
)
from .config import get_llm_for_node
from .models import AgentState, IntentLiteral

logger = logging.getLogger(__name__)


# Инициализация цепочек лениво, чтобы не падать при импорте без ключа
_CHAINS: Dict[str, Runnable] | None = None


def ensure_chains() -> Dict[str, Runnable]:
    """
    Создаем цепочки при первом обращении.
    Ошибки инициализации пробрасываем в вызывающий код для graceful деградации.
    """
    global _CHAINS
    if _CHAINS is not None:
        return _CHAINS
    try:
        _CHAINS = {
            "intent": build_intent_chain(get_llm_for_node("intent")),
            "search": build_search_chain(get_llm_for_node("search")),
            "summary": build_summary_chain(get_llm_for_node("summarize")),
            "sentiment": build_sentiment_chain(get_llm_for_node("sentiment")),
            "fallback": build_fallback_chain(get_llm_for_node("fallback")),
        }
        return _CHAINS
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось инициализировать LLM/цепочки: %s", exc)
        raise


def normalize_intents(raw: list[IntentLiteral], user_query: str) -> list[IntentLiteral]:
    """
    Оставляем допустимые интенты, убираем дубликаты.
    Сохраняем только намерения из заданного списка и обеспечиваем порядок: search перед summarize.
    """
    seen: set[str] = set()
    intents: list[IntentLiteral] = []
    for intent in raw:
        if intent not in ("search", "summarize", "sentiment", "none"):
            continue
        if intent in seen:
            continue
        seen.add(intent)
        intents.append(intent)

    try:
        search_idx = intents.index("search")
        summarize_idx = intents.index("summarize")
        if search_idx > summarize_idx:
            intents.pop(search_idx)
            intents.insert(summarize_idx, "search")
    except ValueError:
        pass

    if intents:
        return intents
    return ["none"]


def _fetch_plain_text(url: str, max_bytes: int = 200_000, timeout: int = 12) -> str:
    """Скачиваем HTML и приводим к простому тексту без тегов."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp:
            if resp.info().get_content_maintype() != "text":
                return ""
            raw = resp.read(max_bytes)
        decoded = raw.decode("utf-8", errors="ignore")
        text = re.sub(r"(?s)<script.*?>.*?</script>", " ", decoded)
        text = re.sub(r"(?s)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        logger.debug("Ошибка загрузки %s: %s", url, exc)
        return ""


def bump_index(state: AgentState) -> Dict[str, int]:
    """Увеличиваем указатель текущего интента."""
    return {"next_intent_index": state.next_intent_index + 1}


def append_step_result(state: AgentState, label: str, payload: str) -> str:
    """Копим результаты шагов в финальном ответе."""
    clean_payload = payload.replace("**", "")
    parts: list[str] = []
    if state.final_answer:
        parts.append(state.final_answer)
    parts.append(f"{label}:\n{clean_payload}")
    return "\n\n".join(parts)


# --- Ноды графа ---

def intent_recognition_node(state: AgentState) -> Dict:
    """Распознаем интенты в пользовательском запросе."""
    try:
        chains = ensure_chains()
        prediction = chains["intent"].invoke({"user_query": state.user_query})
        intents = normalize_intents(prediction.intents, state.user_query)
        logger.info("Определены интенты: %s", intents)
        return {
            "detected_intents": intents,
            "intermediate_result": state.user_query,
            "final_answer": None,
            "next_intent_index": 0,
        }
    except Exception as exc:  # noqa: BLE001
        error_msg = (
            "Не удалось определить интенты (проверьте ключ/квоты OpenRouter). "
            "Детали: "
            f"{exc}"
        )
        logger.error(error_msg)
        return {
            "detected_intents": ["none"],
            "intermediate_result": error_msg,
            "final_answer": error_msg,
            "next_intent_index": 0,
        }


def router_node(state: AgentState) -> AgentState:
    """Передаем состояние в маршрутизацию без изменений."""
    return state


def search_tool_handler(state: AgentState) -> Dict:
    """Реальный поиск через DuckDuckGo с фолбэком на LLM."""
    query = state.intermediate_result or state.user_query

    results_lines: list[str] = []
    urls: list[str] = []
    try:
        from ddgs import DDGS  # локальный импорт, чтобы не тянуть при старте

        with DDGS() as ddg:
            for item in ddg.text(query, max_results=5):
                title = item.get("title") or "Без названия"
                url = item.get("href") or ""
                if url:
                    urls.append(url)
                snippet = item.get("body") or ""
                results_lines.append(f"- {title} ({url}): {snippet}")
        payload = "\n".join(results_lines)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка DuckDuckGo: %s", exc)
        payload = ""

    if not payload:
        try:
            chains = ensure_chains()
            results = chains["search"].invoke({"query": query})
            lines = [
                f"- {item.title} ({item.url}): {item.snippet}"
                for item in results.results
            ]
            payload = "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка псевдо-поиска: %s", exc)
            payload = (
                "Не удалось получить результаты поиска.\n"
                f"- Запрос: {query}"
            )
            urls = []

    full_payload = payload

    # Глубокий режим: скачиваем страницы и суммаризируем
    if state.search_mode == "deep":
        deep_blocks: list[str] = []
        if urls:
            try:
                chains = ensure_chains()
            except Exception as exc:  # noqa: BLE001
                logger.error("Не удалось инициализировать LLM для глубокого поиска: %s", exc)
                chains = None

            for url in urls[:2]:
                page_text = _fetch_plain_text(url)
                if not page_text:
                    continue
                if chains is None:
                    continue
                try:
                    summary = chains["summary"].invoke({"content": page_text[:6000]})
                    deep_blocks.append(f"{url}\n{summary.content}")
                except Exception as exc:  # noqa: BLE001
                    logger.error("Ошибка суммаризации страницы %s: %s", url, exc)
                    continue
        if deep_blocks:
            deep_section = "Сводки страниц:\n" + "\n\n".join(deep_blocks)
            full_payload = payload + "\n\n" + deep_section if payload else deep_section

    updated = {
        "intermediate_result": full_payload,
        "final_answer": append_step_result(state, "Поиск", full_payload),
    }
    updated.update(bump_index(state))
    return updated


def summarize_handler(state: AgentState) -> Dict:
    """Суммаризация текущего промежуточного результата."""
    content = state.intermediate_result or state.user_query
    try:
        chains = ensure_chains()
        summary = chains["summary"].invoke({"content": content})
        payload = str(summary.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка суммаризации: %s", exc)
        payload = "Не удалось сделать краткое изложение."

    updated = {
        "intermediate_result": payload,
        "final_answer": append_step_result(state, "Суммаризация", payload),
    }
    updated.update(bump_index(state))
    return updated


def sentiment_handler(state: AgentState) -> Dict:
    """Определяем тональность текста."""
    content = state.intermediate_result or state.user_query
    try:
        chains = ensure_chains()
        sentiment = chains["sentiment"].invoke({"content": content})
        payload = str(sentiment.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка тональности: %s", exc)
        payload = "Не удалось определить тональность."

    updated = {
        "intermediate_result": payload,
        "final_answer": append_step_result(state, "Тональность", payload),
    }
    updated.update(bump_index(state))
    return updated


def fallback_handler(state: AgentState) -> Dict:
    """Ответ напрямую через LLM, если интенты не найдены."""
    try:
        chains = ensure_chains()
        reply = chains["fallback"].invoke({"query": state.user_query})
        payload = str(reply.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка fallback-ответа: %s", exc)
        payload = "Не удалось сформировать ответ (проверьте ключ/квоты OpenRouter)."

    updated = {
        "intermediate_result": payload,
        "detected_intents": state.detected_intents or ["none"],
        "final_answer": append_step_result(state, "Ответ", payload),
    }
    updated.update(bump_index(state))
    return updated


def finalize_node(state: AgentState) -> Dict:
    """Формируем финальный ответ."""
    answer = state.final_answer or state.intermediate_result or "Ответ не сформирован."
    return {"final_answer": answer}
