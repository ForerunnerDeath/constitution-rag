from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    source_path: Path = Path("data/raw/constitution.txt")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "constitution_e5_small"

    embedding_model: str = "intfloat/multilingual-e5-small"
    min_score: float = 0.70

    llm_enabled: bool = False
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
