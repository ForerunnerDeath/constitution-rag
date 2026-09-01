from pydantic import BaseModel, ConfigDict, Field

API_DISCLAIMER = (
    "Сервис возвращает выдержки из текста Конституции РФ "
    "и не является юридической консультацией."
)


class SearchHitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    quote: str
    ref: str
    article: str | None
    part: int | None
    part_label: str | None
    score: float | None


class SearchResponse(BaseModel):
    hits: list[SearchHitResponse]
    took_ms: float
    collection_version: str
    disclaimer: str = API_DISCLAIMER


class ArticleChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    quote: str
    ref: str
    part: int | None
    part_label: str | None


class ArticleResponse(BaseModel):
    article: str
    chunks: list[ArticleChunkResponse]
    disclaimer: str = API_DISCLAIMER


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = False


class CitationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    quote: str
    ref: str
    article: str | None
    part: int | None
    part_label: str | None


class AskResponse(BaseModel):
    found: bool
    answer: str | None
    message: str | None
    citations: list[CitationResponse]
    llm_used: bool
    disclaimer: str = API_DISCLAIMER
