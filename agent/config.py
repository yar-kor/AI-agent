from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Список кодов ошибок, при которых имеет смысл переключаться на следующую модель
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Загружаем переменные из .env
load_dotenv()

logger = logging.getLogger(__name__)

# Дефолтная конфигурация LLMов и параметров сэмплинга (бесплатные модели OpenRouter)
DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "defaults": {
        "model": "openai/gpt-oss-20b:free",
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
    "nodes": {
        "intent": {
            "models": [
                {"model": "openai/gpt-oss-20b:free", "temperature": 0.0, "max_tokens": 256},
                {"model": "mistralai/mistral-7b-instruct:free", "temperature": 0.0, "max_tokens": 256},
                {"model": "meta-llama/llama-3.2-3b-instruct:free", "temperature": 0.1, "max_tokens": 256},
            ]
        },
        "search": {
            "models": [
                {"model": "openai/gpt-oss-20b:free", "temperature": 0.25, "max_tokens": 512},
                {"model": "mistralai/mistral-7b-instruct:free", "temperature": 0.2, "max_tokens": 512},
            ]
        },
        "summarize": {
            "models": [
                {"model": "openai/gpt-oss-20b:free", "temperature": 0.25, "max_tokens": 768},
                {"model": "mistralai/mistral-7b-instruct:free", "temperature": 0.25, "max_tokens": 768},
                {"model": "moonshotai/kimi-k2:free", "temperature": 0.4, "max_tokens": 768},
            ]
        },
        "sentiment": {
            "models": [
                {"model": "openai/gpt-oss-20b:free", "temperature": 0.1, "max_tokens": 256},
                {"model": "mistralai/mistral-7b-instruct:free", "temperature": 0.1, "max_tokens": 256},
                {"model": "meta-llama/llama-3.2-3b-instruct:free", "temperature": 0.1, "max_tokens": 256},
            ]
        },
        "fallback": {
            "models": [
                {"model": "openai/gpt-oss-20b:free", "temperature": 0.6, "max_tokens": 1024},
                {"model": "mistralai/mistral-7b-instruct:free", "temperature": 0.6, "max_tokens": 1024},
                {"model": "moonshotai/kimi-k2:free", "temperature": 0.7, "max_tokens": 1024},
                {"model": "meta-llama/llama-3.2-3b-instruct:free", "temperature": 0.6, "max_tokens": 1024},
                {"model": "openai/gpt-oss-120b:free", "temperature": 0.6, "max_tokens": 1024},
            ]
        },
    },
}


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """Читаем конфиг LLMов из yaml, при ошибке возвращаем пустой dict."""
    if not path.exists():
        logger.warning("Файл конфигурации LLM %s не найден, используем дефолты.", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Корень конфига должен быть объектом.")
            return data
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Не удалось прочитать конфиг LLM %s (%s), используем дефолты.", path, exc
        )
        return {}


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Поверх дефолтов накладываем пользовательский конфиг."""
    config = deepcopy(base)
    if override.get("defaults"):
        for key, value in override["defaults"].items():
            config["defaults"][key] = value

    for node, node_cfg in (override.get("nodes") or {}).items():
        if not isinstance(node_cfg, dict):
            continue
        models = node_cfg.get("models")
        if isinstance(models, list) and models:
            config.setdefault("nodes", {})[node] = {"models": models}
    return config


@lru_cache(maxsize=1)
def get_llm_config() -> Dict[str, Any]:
    """Возвращает итоговый конфиг LLMов (дефолт + пользовательский)."""
    path = Path(os.getenv("LLM_CONFIG_PATH", "llm_config.yaml"))
    override = _load_yaml_config(path)
    return _merge_config(DEFAULT_LLM_CONFIG, override)


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


@lru_cache(maxsize=64)
def _build_chat_llm(
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    base_url: str,
    referer: str | None,
    title: str | None,
) -> ChatOpenAI:
    """Создает и кэширует клиент ChatOpenAI под OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Переменная окружения OPENROUTER_API_KEY не найдена. "
            "Добавьте ее в .env или окружение с вашим ключом OpenRouter."
        )

    headers = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        default_headers=headers or None,
    )


def get_llm_for_node(node: str) -> ChatOpenAI:
    """
    Возвращает ChatOpenAI для заданной ноды по конфигу (с приоритетным списком моделей).
    Перебирает модели по порядку до первой успешно инициализированной.
    Если вызов клиента вернет retryable ошибку, вызывающая сторона должна пробовать следующую модель.
    """
    config = get_llm_config()
    defaults = config.get("defaults", {})
    nodes_cfg = config.get("nodes", {})
    node_cfg = nodes_cfg.get(node) or {}
    models: List[Dict[str, Any]] = node_cfg.get("models") or []

    if not models:
        fallback_model = defaults.get("model")
        logger.warning(
            "Для ноды %s не заданы модели, используем дефолтную %s", node, fallback_model
        )
        models = [{"model": fallback_model}]

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    referer = os.getenv("OPENROUTER_REFERER")
    title = os.getenv("OPENROUTER_TITLE")

    last_error: Exception | None = None
    for model_entry in models:
        merged = {**defaults, **(model_entry or {})}
        model_name = merged.get("model")
        if not model_name:
            continue

        temperature = _as_float(merged.get("temperature"), _as_float(defaults.get("temperature"), 0.3))
        top_p = _as_float(merged.get("top_p"), _as_float(defaults.get("top_p"), 1.0))
        max_tokens = _as_int(merged.get("max_tokens"), _as_int(defaults.get("max_tokens"), 2048))

        try:
            llm = _build_chat_llm(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                base_url=base_url,
                referer=referer,
                title=title,
            )
            logger.info("Нода %s использует модель %s", node, model_name)
            return llm
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error("Не удалось инициализировать модель %s для ноды %s: %s", model_name, node, exc)
            continue

    raise RuntimeError(
        f"Не удалось подобрать ни одну модель для ноды {node}. Последняя ошибка: {last_error}"
    )
