
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float
    ollama_base_url: str
    openrouter_api_key: str
    openrouter_base_url: str


def load_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider not in {"ollama", "openrouter"}:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Choose 'ollama' or 'openrouter'."
        )

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "mistral")
    else:
        model = os.getenv(
            "OPENROUTER_MODEL",
            "openai/gpt-4o-mini",
        )

    return LLMConfig(
        provider=provider,
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        openrouter_api_key=os.getenv(
            "OPENROUTER_API_KEY",
            "",
        ),  
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
    )
