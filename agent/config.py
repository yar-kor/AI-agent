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

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "defaults": {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 1024,
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "api_key_env": "GROQ_API_KEY",
    },
    "nodes": {
        # Приоритет: GigaChat (Cloud.ru) -> Groq llama-3.1-8b
        "intent": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "base_url": os.getenv("CLOUDRU_BASE_URL", "https://foundation-models.api.cloud.ru/v1"),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {"model": "llama-3.1-8b-instant", "temperature": 0.0, "max_tokens": 256},
            ]
        },
        "search": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.25,
                    "max_tokens": 640,
                    "base_url": os.getenv("CLOUDRU_BASE_URL", "https://foundation-models.api.cloud.ru/v1"),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {"model": "llama-3.1-8b-instant", "temperature": 0.2, "max_tokens": 640},
            ]
        },
        "summarize": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "base_url": os.getenv("CLOUDRU_BASE_URL", "https://foundation-models.api.cloud.ru/v1"),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {"model": "llama-3.1-8b-instant", "temperature": 0.3, "max_tokens": 1024},
            ]
        },
        "sentiment": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.1,
                    "max_tokens": 256,
                    "base_url": os.getenv("CLOUDRU_BASE_URL", "https://foundation-models.api.cloud.ru/v1"),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {"model": "llama-3.1-8b-instant", "temperature": 0.1, "max_tokens": 256},
            ]
        },
        "fallback": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.55,
                    "max_tokens": 1200,
                    "base_url": os.getenv("CLOUDRU_BASE_URL", "https://foundation-models.api.cloud.ru/v1"),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {"model": "llama-3.1-8b-instant", "temperature": 0.6, "max_tokens": 1200},
            ]
        },
    },
}


def _load_yaml_config(path: Path) -> Dict[str, Any]:
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
    """Итоговый конфиг LLM (дефолтный конфиг + пользовательский YAML)."""
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


def _build_chat_llm(
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    base_url: str,
    api_key: str,
) -> ChatOpenAI:
    if not api_key:
        raise RuntimeError(
            "API ключ не найден. Проверьте переменные окружения для выбранного провайдера."
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def iter_llms_for_node(node: str):
    """
    Итератор по моделям для ноды в порядке приоритета.
    Позволяет переключиться на следующую модель при ошибках.
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

    preferred = os.getenv("LLM_PREFERRED_MODEL")
    if preferred:
        prioritized = []
        rest = []
        for entry in models:
            if entry.get("model") == preferred:
                prioritized.append(entry)
            else:
                rest.append(entry)
        models = prioritized + rest

    for model_entry in models:
        merged = {**defaults, **(model_entry or {})}
        model_name = merged.get("model")
        if not model_name:
            continue

        temperature = _as_float(
            merged.get("temperature"), _as_float(defaults.get("temperature"), 0.3)
        )
        top_p = _as_float(merged.get("top_p"), _as_float(defaults.get("top_p"), 1.0))
        max_tokens = _as_int(merged.get("max_tokens"), _as_int(defaults.get("max_tokens"), 2048))
        base_url = merged.get("base_url") or defaults.get("base_url") or "https://api.groq.com/openai/v1"
        api_key_env = merged.get("api_key_env") or defaults.get("api_key_env") or "GROQ_API_KEY"
        api_key = os.getenv(api_key_env)

        try:
            llm = _build_chat_llm(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
            )
            yield model_name, llm
        except Exception as exc:  
            logger.error("Не удалось инициализировать модель %s для ноды %s: %s", model_name, node, exc)
            continue
