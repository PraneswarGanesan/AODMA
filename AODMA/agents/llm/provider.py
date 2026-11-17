
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agents.llm.config import load_llm_config


def get_llm() -> BaseChatModel:
    config = load_llm_config()

    if config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
        )

    if config.provider == "openrouter":
        if not config.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing from your .env file."
            )

        return ChatOpenAI(
            model=config.model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=config.temperature,
        )

    raise ValueError(
        f"Unsupported provider: {config.provider}"
    )
