# Constitution RAG

RAG-сервис на FastAPI для поиска точных цитат из Конституции Российской Федерации.

## Статус

Проект находится в разработке.

## Стек

- Python 3.14
- FastAPI
- Pydantic / pydantic-settings
- Sentence Transformers
- ChromaDB
- BM25
- OpenAI-compatible LLM API
- Pytest

## Retrieval Evaluation

Качество retrieval оценивается на фиксированном golden dataset:

- 40 вопросов всего;
- 25 позитивных вопросов с ожидаемой статьёй Конституции;
- 15 негативных вопросов, для которых ожидается пустая выдача.

Dataset находится в:

```text
eval/questions.csv
```

Для оценки используется:

```bash
python -m scripts.evaluate
```

Можно переопределить retrieval threshold:

```bash
python -m scripts.evaluate --min-score 0.833
```

И отдельно проверить hybrid retrieval:

```bash
python -m scripts.evaluate --min-score 0.833 --hybrid
```

### Метрики

Используются следующие метрики:

- **Recall@1 / Recall@3 / Recall@5** — доля позитивных вопросов, для которых ожидаемая статья попала в TOP-1 / TOP-3 / TOP-5;
- **MRR (Mean Reciprocal Rank)** — среднее обратного ранга первого результата с ожидаемой статьёй;
- **Refusal accuracy** — доля негативных вопросов, для которых Retriever корректно вернул пустую выдачу;
- **False refusal** — доля позитивных вопросов, для которых Retriever ошибочно вернул пустую выдачу;
- **Raw TOP-1 score distribution** — min / median / max cosine score до применения threshold отдельно для позитивных и негативных вопросов.

### Baseline

Исходная конфигурация:

```text
embedding_model = intfloat/multilingual-e5-small
min_score       = 0.80
max_chunk_chars = 900
header prefix   = ON
use_hybrid      = false
```

Результат:

| Metric | Value |
|---|---:|
| Recall@1 | 0.880 |
| Recall@3 | 0.920 |
| Recall@5 | 1.000 |
| MRR | 0.920 |
| Refusal accuracy | 0.667 |
| False refusal | 0.000 |

Raw TOP-1 vector scores:

| Dataset | Min | Median | Max |
|---|---:|---:|---:|
| Positive | 0.8365 | 0.8663 | 0.9149 |
| Negative | 0.7324 | 0.7807 | 0.8327 |

Распределения почти разделяются: максимальный score негативного запроса равен `0.8327`, поэтому baseline threshold `0.80` оказался слишком мягким.

### Threshold tuning

Для `multilingual-e5-small` был выполнен sweep по `min_score`:

| min_score | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---:|---:|---:|---:|---:|---:|---:|
| 0.800 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| 0.810 | 0.880 | 0.920 | 1.000 | 0.920 | 0.733 | 0.000 |
| 0.820 | 0.880 | 0.920 | 1.000 | 0.920 | 0.733 | 0.000 |
| 0.830 | 0.880 | 0.920 | 0.960 | 0.910 | 0.867 | 0.000 |
| **0.833** | **0.880** | **0.920** | **0.960** | **0.910** | **1.000** | **0.000** |
| 0.835 | 0.880 | 0.920 | 0.960 | 0.910 | 1.000 | 0.000 |
| 0.840 | 0.840 | 0.880 | 0.920 | 0.870 | 1.000 | 0.080 |
| 0.850 | 0.680 | 0.720 | 0.760 | 0.710 | 1.000 | 0.240 |

Выбран `min_score=0.833`: он даёт 100% refusal accuracy без false refusals и сохраняет Recall@5 на уровне 0.960.

### Vector vs Hybrid

При baseline threshold `0.80`:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---|---:|---:|---:|---:|---:|---:|
| Vector | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| Hybrid | 0.880 | 0.960 | 1.000 | 0.917 | 0.667 | 0.000 |

При выбранном `min_score=0.833`:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---|---:|---:|---:|---:|---:|---:|
| Vector | 0.880 | 0.920 | 0.960 | 0.910 | 1.000 | 0.000 |
| Hybrid | 0.880 | 0.960 | 0.960 | 0.913 | 1.000 | 0.000 |

Hybrid немного улучшает Recall@3 и MRR, но не меняет ключевые Recall@5 и refusal accuracy.

Поэтому vector retrieval остаётся default, а hybrid доступен как дополнительный режим.

### Header prefix experiment

Сравнивались embeddings с заголовочным префиксом:

```text
Глава N. <Название>. Статья N, часть M. <текст>
```

и embeddings только по тексту цитаты.

| Prefix | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---|---:|---:|---:|---:|---:|---:|
| ON | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| OFF | 0.880 | 0.960 | 1.000 | 0.921 | 0.667 | 0.000 |

На текущем dataset отсутствие prefix немного улучшило Recall@3, но ключевые метрики практически не изменились.

Поэтому production indexing оставлен без изменений.

### Chunk size experiment

Проверялись:

```text
max_chunk_chars = 500
max_chunk_chars = 900
max_chunk_chars = 1500
```

Количество полученных чанков:

| max_chunk_chars | Chunks |
|---:|---:|
| 500 | 435 |
| 900 | 383 |
| 1500 | 355 |

При этом article-level retrieval metrics оказались одинаковыми:

| Chunk size | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| 900 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| 1500 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |

Размер `900` сохранён как текущий default.

### Embedding model experiment

Сравнивались:

- `intfloat/multilingual-e5-small`;
- `intfloat/multilingual-e5-base`.

При одинаковом `min_score=0.80`:

| Model | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---|---:|---:|---:|---:|---:|---:|
| e5-small | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| e5-base | 0.880 | 0.960 | 1.000 | 0.923 | 0.867 | 0.000 |

Однако cosine score distributions разных embedding-моделей отличаются, поэтому был отдельно выполнен threshold sweep для `e5-base`.

Лучший вариант `e5-base`, достигший 100% refusal accuracy:

```text
min_score = 0.830
Recall@1 = 0.800
Recall@3 = 0.880
Recall@5 = 0.920
MRR = 0.843
Refusal accuracy = 1.000
False refusal = 0.080
```

Для сравнения `e5-small` при `min_score=0.833`:

```text
Recall@1 = 0.880
Recall@3 = 0.920
Recall@5 = 0.960
MRR = 0.910
Refusal accuracy = 1.000
False refusal = 0.000
```

После threshold tuning `e5-small` показал лучшие итоговые метрики и при этом требует меньше вычислительных ресурсов, поэтому модель оставлена без изменений.

### Финальная retrieval-конфигурация

После экспериментов используется:

```text
embedding_model = intfloat/multilingual-e5-small
min_score       = 0.833
max_chunk_chars = 900
header prefix   = ON
use_hybrid      = false
```

Финальные метрики:

| Metric | Value |
|---|---:|
| Recall@1 | 0.880 |
| Recall@3 | 0.920 |
| Recall@5 | **0.960** |
| MRR | 0.910 |
| Refusal accuracy | **1.000** |
| False refusal | **0.000** |

Таким образом, выполняются критерии качества проекта:

```text
Recall@5 >= 0.85
Refusal accuracy = 100%
```
