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
- Docker / Docker Compose

## Retrieval Evaluation

Evaluation разделена на два независимых набора:

- `eval/dev.csv` — 40 вопросов: 25 positive и 15 negative;
- `eval/holdout.csv` — 40 новых вопросов: 20 positive и 20 negative.

`dev` используется для разработки и выбора retrieval-конфигурации:

- подбора `min_score`;
- сравнения embedding-моделей;
- выбора размера chunk;
- проверки header prefix;
- сравнения vector и hybrid retrieval.

Изначальный golden dataset проекта стал `dev`, поскольку результаты экспериментов по нему уже были известны.

`holdout` был составлен после выбора финальной retrieval-конфигурации и не использовался для подбора `min_score` или других параметров. Он применяется только для независимой проверки переноса выбранной конфигурации на новые вопросы.

Для оценки dev-набора используется:

```bash
python -m scripts.evaluate --dataset eval/dev.csv
```

Можно переопределить retrieval threshold:

```bash
python -m scripts.evaluate --dataset eval/dev.csv --min-score 0.833
```

И отдельно проверить hybrid retrieval:

```bash
python -m scripts.evaluate --dataset eval/dev.csv --min-score 0.833 --hybrid
```

Для независимой финальной оценки используется:

```bash
python -m scripts.evaluate --dataset eval/holdout.csv
```

После первого прогона `holdout` его результаты не использовались для повторной настройки `min_score`.
```

### Метрики

Используются следующие метрики:

- **Recall@1 / Recall@3 / Recall@5** — доля позитивных вопросов, для которых ожидаемая статья попала в TOP-1 / TOP-3 / TOP-5;
- **MRR (Mean Reciprocal Rank)** — среднее обратного ранга первого результата с ожидаемой статьёй;
- **Refusal accuracy** — доля негативных вопросов, для которых Retriever корректно вернул пустую выдачу;
- **False refusal** — доля позитивных вопросов, для которых Retriever ошибочно вернул пустую выдачу;
- **Raw TOP-1 score distribution** — min / median / max cosine score до применения threshold отдельно для позитивных и негативных вопросов.

Все эксперименты ниже выполнялись на `eval/dev.csv`. Этот набор используется для настройки и сравнения retrieval-конфигураций, поэтому полученные на нём значения являются dev/in-sample метриками, а не независимой финальной оценкой качества.

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

### Dev-метрики выбранной конфигурации

На `eval/dev.csv` выбранная конфигурация показывает:

| Metric | Value |
|---|---:|
| Recall@1 | 0.880 |
| Recall@3 | 0.920 |
| Recall@5 | 0.960 |
| MRR | 0.910 |
| Refusal accuracy | 1.000 |
| False refusal | 0.000 |

Raw TOP-1 cosine scores:

| Class | Min | Median | Max |
|---|---:|---:|---:|
| Positive | 0.8365 | 0.8663 | 0.9149 |
| Negative | 0.7324 | 0.7807 | 0.8327 |

На dev-наборе threshold `0.833` полностью разделяет positive и negative запросы.

### Финальная holdout evaluation

После выбора retrieval-конфигурации был выполнен независимый прогон на `eval/holdout.csv`.

До этого прогона holdout-вопросы не использовались для настройки `min_score`, выбора модели, chunk size, header prefix или режима retrieval.

Результат первого holdout-прогона:

| Metric | Value |
|---|---:|
| Recall@1 | 0.950 |
| Recall@3 | 0.950 |
| Recall@5 | 1.000 |
| MRR | 0.963 |
| Refusal accuracy | 0.900 |
| False refusal | 0.000 |

Raw TOP-1 cosine scores:

| Class | Min | Median | Max |
|---|---:|---:|---:|
| Positive | 0.8583 | 0.8904 | 0.9079 |
| Negative | 0.7892 | 0.8136 | 0.8685 |

На holdout-наборе распределения positive и negative scores уже пересекаются:

```text
min positive = 0.8583
max negative = 0.8685
```

При текущем `min_score=0.833` все 20 positive-вопросов проходят threshold, но 2 из 20 negative-вопросов также получают непустую выдачу:

```text
Refusal accuracy = 18 / 20 = 0.900
False refusal    = 0 / 20  = 0.000
```

Таким образом, прежний результат `Refusal accuracy = 1.000` корректно описывает dev-набор, на котором подбирался threshold, но не должен трактоваться как независимая оценка качества на новых запросах.

После получения holdout-результатов `min_score=0.833` по этому набору не перенастраивался.

Независимая holdout-оценка подтверждает высокий retrieval recall:

```text
Recall@5 = 1.000
```

и одновременно показывает ограничение текущего абсолютного cosine threshold на более сложных negative-запросах:

```text
Refusal accuracy = 0.900
```


## Docker

Приложение можно полностью запустить через Docker Compose.

Embedding-модель `intfloat/multilingual-e5-small` загружается во время сборки Docker image и после этого доступна контейнеру без обращения к Hugging Face Hub во время runtime.

Контейнер приложения запускается от non-root пользователя.

### Подготовка

Создайте локальный `.env` на основе:

```text
.env.example
```

Исходный текст Конституции должен находиться по пути:

```text
data/raw/constitution.txt
```

Этот каталог подключается к контейнеру read-only.

Chroma index хранится отдельно в Docker named volume и не теряется при пересоздании контейнера.

### Сборка image

```bash
docker compose build
```

Во время первой сборки устанавливаются Python-зависимости и скачивается embedding-модель, поэтому первый build может занять несколько минут.

Последующие сборки используют Docker layer cache.

### Ingest

Перед первым запуском API необходимо построить Chroma index:

```bash
docker compose run --rm app python -m scripts.ingest --recreate
```

Ожидаемый результат:

```text
Units: 384
Chunks: 383
Vectors: 383
Stored: 383
```

Ingest выполняется во временном контейнере.

После его завершения контейнер удаляется, но индекс сохраняется в Docker volume.

### Запуск API

```bash
docker compose up -d
```

Проверить состояние контейнера:

```bash
docker compose ps
```

После успешного старта контейнер должен перейти в состояние:

```text
healthy
```

API доступен по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Health check

```bash
curl http://127.0.0.1:8000/healthz
```

Ответ:

```json
{
  "status": "ok"
}
```

Docker image также содержит встроенный healthcheck этого endpoint.

### Readiness check

```bash
curl http://127.0.0.1:8000/readyz
```

После успешного ingest:

```json
{
  "status": "ok",
  "stored": 383
}
```

`/healthz` показывает, что само приложение работает.

`/readyz` дополнительно проверяет, что Chroma collection содержит данные и сервис готов выполнять retrieval.

### Search

Пример semantic search:

```bash
curl -G "http://127.0.0.1:8000/search" \
  --data-urlencode "q=Кто является источником власти?" \
  --data-urlencode "k=5"
```

Ответ содержит найденные цитаты, ссылки на статьи и cosine scores.

### Получение статьи

```bash
curl http://127.0.0.1:8000/articles/3
```

Endpoint возвращает все сохранённые chunks указанной статьи в порядке документа.

### RAG-вопрос

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Кто является источником власти?",
    "k": 5
  }'
```

Если:

```env
LLM_ENABLED=false
```

retrieval всё равно выполняется и citations возвращаются, но генерация ответа отключена:

```text
found = true
answer = null
llm_used = false
```

### Использование Ollama с Docker

Ollama не входит в текущий Docker Compose stack.

Если Ollama работает на host-машине, адрес:

```text
http://127.0.0.1:11434
```

из Docker-контейнера указывает уже на сам контейнер, а не на host.

При использовании Docker Desktop для обращения к Ollama на host можно настроить:

```env
LLM_ENABLED=true
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:4b-instruct
```

### Логи

```bash
docker compose logs -f app
```

Логи содержат:

- `request_id`;
- HTTP method/path/status/duration;
- sanitized question для `/search` и `/ask`;
- найденные chunk IDs и scores;
- `embed_ms`;
- `search_ms`;
- `llm_ms`;
- `prompt_version`;
- факт вызова LLM;
- refusal flag.

### Persistence Chroma

Chroma хранится в Docker named volume:

```text
chroma_data
```

Поэтому:

```bash
docker compose down
docker compose up -d
```

не удаляет построенный индекс.

Для намеренного удаления контейнеров **и volumes** используется:

```bash
docker compose down -v
```

После этой команды Chroma index будет удалён и потребуется повторный ingest.

### Остановка

```bash
docker compose down
```
