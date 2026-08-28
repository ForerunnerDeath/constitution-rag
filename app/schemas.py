from pydantic import BaseModel, ConfigDict


class SearchHitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    quote: str
    ref: str
    article: str | None
    part: int | None
    part_label: str | None
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHitResponse]
    took_ms: float
    collection_version: str


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
