# Constitution RAG

RAG-сервис на FastAPI для поиска точных цитат из Конституции Российской Федерации и опциональной генерации ответа строго по найденным фрагментам.

## Возможности

- загрузка и нормализация исходного текста Конституции РФ;
- структурный парсинг по главам, статьям и частям;
- chunking с устойчивыми идентификаторами;
- embeddings через Sentence Transformers;
- persistent vector index в ChromaDB;
- vector retrieval с query-level relevance gate;
- опциональный hybrid retrieval: vector search + BM25 + Reciprocal Rank Fusion;
- получение всех chunks конкретной статьи;
- RAG-ответ через OpenAI-compatible LLM API;
- отказ от генерации при недостаточном контексте;
- citations только из реально retrieved документов;
- request ID, structured logs и latency-метрики;
- per-IP in-memory rate limiting;
- corpus checksum и deterministic index revision;
- Docker / Docker Compose;
- retrieval и generation evaluation;
- CI с Ruff, Pyright и Pytest.

## Стек

- Python 3.14
- FastAPI
- Pydantic / pydantic-settings
- Sentence Transformers
- ChromaDB
- BM25
- OpenAI-compatible LLM API
- Pytest
- Ruff
- Pyright
- Docker / Docker Compose
- GitHub Actions

## Конфигурация

Базовые настройки задаются через `.env`. Пример находится в `.env.example`.

| Переменная | Default | Назначение |
|---|---|---|
| `SOURCE_PATH` | `data/raw/constitution.txt` | путь к исходному тексту Конституции |
| `CHROMA_PATH` | `data/chroma` | каталог persistent Chroma |
| `CHROMA_COLLECTION` | `constitution_e5_small` | имя Chroma collection |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | embedding-модель |
| `MIN_SCORE` | `0.833` | query-level cosine relevance threshold |
| `LLM_ENABLED` | `false` | включить генерацию ответа |
| `LLM_BASE_URL` | пусто | base URL OpenAI-compatible API |
| `LLM_API_KEY` | пусто | API key |
| `LLM_MODEL` | пусто | имя LLM |
| `LLM_MAX_TOKENS` | `512` | maximum generation tokens |
| `LLM_TIMEOUT_SECONDS` | `20` | timeout LLM-запроса |
| `RATE_LIMIT_PER_MINUTE` | `60` | лимит запросов на клиента |

## Исходный корпус и целостность индекса

Проект не скачивает и не обновляет текст Конституции автоматически. Подготовленный исходный файл должен находиться по пути:

```text
data/raw/constitution.txt
```

Обновление корпуса выполняется явно: файл заменяется пользователем, после чего запускается ingest.

При ingest сервис вычисляет SHA-256 исходного корпуса и сохраняет его в metadata Chroma collection как `corpus_checksum`.

Дополнительно вычисляется deterministic `index_revision`. Она зависит от:

- SHA-256 исходного корпуса;
- embedding-модели;
- размерности embeddings;
- фактически построенных chunks и их содержимого.

Поэтому одинаковый индекс при повторной сборке получает ту же revision, а изменение корпуса, embedding-конфигурации или результата chunking меняет revision.

Перед изменением существующего индекса integrity metadata инвалидируется. Новая `index_revision` и затем `corpus_checksum` записываются только после успешного завершения ingest и проверки количества сохранённых chunks. `corpus_checksum` записывается последним и служит финальным маркером завершённой индексации.

На startup приложение проверяет:

1. совместимость embedding-модели и размерности с metadata индекса;
2. совпадение текущего исходного корпуса с сохранённым `corpus_checksum`;
3. наличие `index_revision` у непустого индекса.

Если исходный файл изменился, индекс был создан старой версией приложения без необходимых metadata или предыдущий ingest завершился некорректно, приложение не использует такой индекс и требует повторного ingest.

После обновления с версии проекта, где `corpus_checksum` или `index_revision` ещё не сохранялись, рекомендуется один раз пересобрать индекс:

```bash
python -m scripts.ingest --recreate
```

Для Docker:

```bash
docker compose run --rm app python -m scripts.ingest --recreate
```

## Локальный запуск

Требуется Python 3.14.

Создать виртуальное окружение и установить зависимости:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Создать `.env` на основе `.env.example`, подготовить `data/raw/constitution.txt`, затем построить индекс:

```bash
python -m scripts.ingest --recreate
```

Ожидаемый результат для текущего корпуса:

```text
Units: 384
Chunks: 383
Vectors: 383
Stored: 383
```

Запуск API:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Docker

Приложение можно полностью запустить через Docker Compose.

Docker image использует Python 3.14. Embedding-модель `intfloat/multilingual-e5-small` загружается во время сборки image и после этого доступна контейнеру без обращения к Hugging Face Hub во время runtime.

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

Каталог с исходным корпусом подключается к контейнеру read-only.

Chroma index хранится отдельно в Docker named volume и не теряется при пересоздании контейнера.

### Сборка image

```bash
docker compose build
```

Во время первой сборки устанавливаются Python-зависимости и скачивается embedding-модель, поэтому первый build может занять несколько минут. Последующие сборки используют Docker layer cache.

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

Ingest выполняется во временном контейнере. После его завершения контейнер удаляется, но индекс сохраняется в Docker volume.

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

Для намеренного удаления контейнеров и volumes используется:

```bash
docker compose down -v
```

После этой команды Chroma index будет удалён и потребуется повторный ingest.

### Остановка

```bash
docker compose down
```

## API

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

`/healthz` показывает, что приложение отвечает. Docker image также использует этот endpoint для встроенного healthcheck.

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

`/readyz` дополнительно проверяет, что Chroma collection содержит данные. Проверки embedding provenance, corpus checksum и index revision выполняются раньше — при startup приложения.

### Search

Vector retrieval используется по умолчанию:

```bash
curl -G "http://127.0.0.1:8000/search" \
  --data-urlencode "q=Кто является источником власти?" \
  --data-urlencode "k=5" \
  --data-urlencode "use_hybrid=false"
```

Hybrid retrieval:

```bash
curl -G "http://127.0.0.1:8000/search" \
  --data-urlencode "q=Кто является источником власти?" \
  --data-urlencode "k=5" \
  --data-urlencode "use_hybrid=true"
```

`use_hybrid=false` использует vector retrieval. При `use_hybrid=true` после успешного vector relevance gate дополнительно запускаются BM25 и Reciprocal Rank Fusion.

`min_score` применяется как **query-level relevance gate**: проверяется cosine score лучшего vector-кандидата. Если TOP-1 ниже `min_score`, Retriever возвращает пустую выдачу и BM25 не запускается. Если TOP-1 проходит threshold, остальные vector-кандидаты не фильтруются повторно тем же `min_score`.

В hybrid-режиме RRF объединяет независимые vector и BM25 candidate pools. Поэтому итоговая выдача может содержать lexical-only hit, которого не было среди vector-кандидатов. Для такого результата `score` может быть `null`: это поле представляет vector cosine similarity, а RRF score намеренно не подменяет его.

Ответ `/search` содержит:

- `hits` — найденные chunks;
- `took_ms` — полное время обработки endpoint;
- `collection_version` — deterministic SHA-256 revision текущего индекса;
- `disclaimer`.

`collection_version` — это не имя Chroma collection. Revision меняется при изменении корпуса, embedding-модели/размерности или фактически построенных chunks.

### Получение статьи

```bash
curl http://127.0.0.1:8000/articles/3
```

Endpoint возвращает все сохранённые chunks указанной статьи в порядке документа.

Параметр `number` принимает номер статьи в формате `N` или `N.M`, например:

```text
3
81
67.1
103.1
```

Некорректный формат, например `foo`, `67.1.2`, `1.` или `-1`, возвращает `422`. Корректный по формату, но отсутствующий номер статьи возвращает `404`.

### RAG-вопрос

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Кто является источником власти?",
    "k": 5,
    "use_hybrid": false
  }'
```

`use_hybrid` имеет ту же семантику, что и в `/search`, и по умолчанию равен `false`.

Поведение `/ask` зависит от retrieval и состояния LLM:

- если Retriever не нашёл достаточно релевантных фрагментов, LLM не вызывается, `found=false`, `answer=null`, `citations=[]`;
- если Retriever нашёл hits, но `LLM_ENABLED=false`, citations возвращаются, `found=true`, `answer=null`, `llm_used=false`;
- если Retriever нашёл hits, но OpenAI-compatible LLM временно недоступна, citations сохраняются, `found=true`, `answer=null`, `llm_used=false`;
- если LLM возвращает отказ `NOT_FOUND`, итоговый ответ имеет `found=false` и `answer=null`, но `citations` могут содержать retrieved фрагменты;
- если LLM формирует обоснованный ответ, `found=true`, `answer` содержит текст ответа, а `citations` формируются только из retrieved hits, а не из свободного вывода модели.

Таким образом, `found` означает наличие обоснованного итогового ответа, а не просто наличие retrieval hits.

Каждый API-ответ также содержит disclaimer о том, что сервис возвращает выдержки из Конституции РФ и не является юридической консультацией.

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

`holdout` был составлен после выбора retrieval-конфигурации и не использовался для подбора `min_score` или других параметров. После первого frozen-прогона holdout не использовался для повторной настройки retrieval.

Для оценки dev-набора:

```bash
python -m scripts.evaluate --dataset eval/dev.csv
```

Можно переопределить retrieval threshold:

```bash
python -m scripts.evaluate --dataset eval/dev.csv --min-score 0.833
```

Hybrid retrieval:

```bash
python -m scripts.evaluate --dataset eval/dev.csv --min-score 0.833 --hybrid
```

Frozen holdout:

```bash
python -m scripts.evaluate --dataset eval/holdout.csv
```

### Метрики

Используются следующие метрики:

- **Recall@1 / Recall@3 / Recall@5** — доля positive-вопросов, для которых ожидаемая статья попала в TOP-1 / TOP-3 / TOP-5;
- **MRR (Mean Reciprocal Rank)** — среднее обратного ранга первого результата с ожидаемой статьёй;
- **Refusal accuracy** — доля negative-вопросов, для которых Retriever корректно вернул пустую выдачу;
- **False refusal** — доля positive-вопросов, для которых Retriever ошибочно вернул пустую выдачу;
- **Raw TOP-1 score distribution** — min / median / max cosine score до применения threshold отдельно для positive и negative запросов.

### Финальная retrieval-конфигурация

Текущий production Retriever использует:

```text
embedding_model = intfloat/multilingual-e5-small
min_score       = 0.833
max_chunk_chars = 900
header prefix   = ON
use_hybrid      = false
relevance gate  = TOP-1 query-level threshold
```

### Dev-метрики текущей конфигурации

Vector retrieval:

| Metric | Value |
|---|---:|
| Recall@1 | 0.880 |
| Recall@3 | 0.920 |
| Recall@5 | 1.000 |
| MRR | 0.920 |
| Refusal accuracy | 1.000 |
| False refusal | 0.000 |

Hybrid retrieval при том же `min_score=0.833`:

| Metric | Value |
|---|---:|
| Recall@1 | 0.880 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| MRR | 0.920 |
| Refusal accuracy | 1.000 |
| False refusal | 0.000 |

Hybrid улучшает Recall@3 на текущем dev-наборе, не ухудшая Recall@5, refusal accuracy и false refusal. Vector retrieval оставлен default, hybrid доступен как опциональный режим.

Raw TOP-1 vector scores на dev:

| Class | Min | Median | Max |
|---|---:|---:|---:|
| Positive | 0.8365 | 0.8663 | 0.9149 |
| Negative | 0.7324 | 0.7807 | 0.8327 |

На dev-наборе threshold `0.833` полностью разделяет positive и negative запросы:

```text
max negative = 0.8327
min positive = 0.8365
```

### Threshold tuning

Первичный sweep показал, что пороги ниже `0.833` пропускают часть negative-запросов, а увеличение threshold выше области разделения начинает повышать риск false refusals.

Для query-level gate ключевые результаты выбора threshold:

| min_score | Refusal accuracy | False refusal |
|---:|---:|---:|
| 0.800 | 0.667 | 0.000 |
| 0.810 | 0.733 | 0.000 |
| 0.820 | 0.733 | 0.000 |
| 0.830 | 0.867 | 0.000 |
| **0.833** | **1.000** | **0.000** |
| 0.835 | 1.000 | 0.000 |
| 0.840 | 1.000 | 0.080 |
| 0.850 | 1.000 | 0.240 |

Выбран `min_score=0.833`: это минимальный из проверенных порогов, который на dev даёт 100% refusal accuracy без false refusals. При текущей query-level семантике его vector retrieval метрики приведены выше: Recall@5 `1.000`, MRR `0.920`.

### Relative relevance gate experiment

Дополнительно была проверена гипотеза о замене абсолютного cosine threshold относительным relevance-критерием.

На `eval/dev.csv` сравнивались:

1. абсолютный TOP-1 cosine score;
2. margin между двумя лучшими результатами: `TOP-1 - TOP-2`;
3. z-score TOP-1 относительно распределения cosine scores по всему корпусу.

Полученные распределения:

| Signal | Positive min | Positive median | Positive max | Negative min | Negative median | Negative max |
|---|---:|---:|---:|---:|---:|---:|
| TOP-1 cosine | 0.8365 | 0.8663 | 0.9149 | 0.7324 | 0.7807 | 0.8327 |
| TOP1-TOP2 margin | 0.0010 | 0.0150 | 0.0547 | 0.0002 | 0.0096 | 0.0190 |
| TOP-1 z-score | 2.7318 | 4.1776 | 6.2747 | 2.5597 | 3.8494 | 5.5486 |

Для `TOP1-TOP2 margin` threshold, отсекающий все negative-запросы, привёл бы примерно к `64%` false refusals среди positive-запросов.

Для TOP-1 z-score аналогичный threshold привёл бы примерно к `88%` false refusals.

Поэтому относительные relevance gates не показали преимущества над откалиброванным абсолютным cosine threshold и не были добавлены в production Retriever.

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

На экспериментальном baseline отсутствие prefix немного улучшило Recall@3, но ключевые метрики практически не изменились. Production indexing оставлен с header prefix.

### Chunk size experiment

Проверялись:

```text
max_chunk_chars = 500
max_chunk_chars = 900
max_chunk_chars = 1500
```

Количество chunks:

| max_chunk_chars | Chunks |
|---:|---:|
| 500 | 435 |
| 900 | 383 |
| 1500 | 355 |

На экспериментальном baseline article-level retrieval metrics оказались одинаковыми:

| Chunk size | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| 900 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| 1500 | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |

Размер `900` сохранён как current default.

### Embedding model experiment

Сравнивались:

- `intfloat/multilingual-e5-small`;
- `intfloat/multilingual-e5-base`.

На экспериментальном baseline при одинаковом `min_score=0.80`:

| Model | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal accuracy | False refusal |
|---|---:|---:|---:|---:|---:|---:|
| e5-small | 0.880 | 0.920 | 1.000 | 0.920 | 0.667 | 0.000 |
| e5-base | 0.880 | 0.960 | 1.000 | 0.923 | 0.867 | 0.000 |

Cosine distributions разных embedding-моделей отличаются, поэтому threshold нельзя переносить между моделями без отдельной калибровки. После экспериментов `multilingual-e5-small` оставлена production-моделью: финальная откалиброванная конфигурация показывает нужный баланс retrieval quality/refusal behavior и требует меньше вычислительных ресурсов.

### Frozen holdout baseline

После выбора retrieval-конфигурации был выполнен один независимый прогон на `eval/holdout.csv`.

Результат первого frozen holdout-прогона:

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

На holdout распределения positive и negative scores пересекаются:

```text
min positive = 0.8583
max negative = 0.8685
```

При `min_score=0.833` в этом frozen-прогоне все 20 positive-вопросов прошли threshold, но 2 из 20 negative-вопросов также получили непустую выдачу:

```text
Refusal accuracy = 18 / 20 = 0.900
False refusal    = 0 / 20  = 0.000
```

После получения holdout-результатов `min_score` по этому набору не перенастраивался.

Важно: после frozen holdout-прогона в Retriever была исправлена семантика threshold — `min_score` стал строго query-level gate вместо фильтрации каждого результата. Holdout после этого намеренно не использовался как итеративный tuning set и повторно не прогонялся. Поэтому числа выше являются зафиксированным независимым baseline на момент первого прогона, а не заявлением о точных метриках текущего commit.

## Generation Evaluation

Качество генерации оценивается отдельно от retrieval.

Используются два набора:

- `eval/generation_dev.csv` — development-набор для исследования поведения генерации;
- `eval/generation_holdout.csv` — независимый набор для финальной проверки выбранной generation-конфигурации.

Evaluator запускает реальный `RAGService`, Retriever, embedding-модель, Chroma и настроенный LLM. Mock LLM для измерения качества генерации не используется.

Для dev-набора:

```bash
python -m scripts.evaluate_generation \
  --dataset eval/generation_dev.csv \
  --output eval/generation_dev_results.json
```

Для frozen holdout:

```bash
python -m scripts.evaluate_generation \
  --dataset eval/generation_holdout.csv \
  --output eval/generation_holdout_results.json
```

JSON-отчёты являются локальными артефактами запусков и не хранятся в Git.

### Автоматические метрики

Evaluator считает:

- **Citation presence rate** — долю сгенерированных ответов, содержащих ссылки;
- **Citation validity / answer** — долю ответов, в которых все найденные ссылки соответствуют `ref` среди retrieved hits;
- **Citation validity / reference** — долю отдельных ссылок, соответствующих `ref` среди retrieved hits.

Автоматическая citation validity проверяет корректность ссылки относительно retrieved sources, но не доказывает, что конкретная ссылка действительно подтверждает утверждение, рядом с которым она поставлена.

Groundedness поэтому оценивается отдельно вручную: каждое существенное утверждение ответа должно подтверждаться текстом в `hits[].quote`.

### Frozen generation holdout baseline

После исследования на `generation_dev` production prompt был возвращён к версии `v1`. После этого generation-конфигурация была заморожена и один раз проверена на ранее не использовавшемся `generation_holdout`.

Конфигурация на момент frozen-прогона:

```text
embedding_model = intfloat/multilingual-e5-small
min_score       = 0.833
k               = 5
llm_model       = qwen3:4b-instruct
prompt_version  = v1
```

Результат на 20 holdout-вопросах:

| Metric | Value |
|---|---:|
| Generated answers | 16 |
| Groundedness | 14/16 = 87.5% |
| Safe insufficient-context refusals | 4/4 = 100% |
| Citation presence | 100% |
| Citation-valid answers | 87.5% |
| Valid citation references | 89.2% |

Holdout не использовался для последующей настройки prompt или retrieval threshold.

Ручная проверка показала, что основные оставшиеся ошибки возникают на сложных synthesis-вопросах: модель иногда слишком широко обобщает смысл retrieved фрагментов. Также валидная ссылка сама по себе не гарантирует semantic entailment конкретного утверждения.

Позднее в retrieval-контуре были внесены внутренние reliability/semantics исправления. Generation holdout после них не использовался для повторной настройки и не объявляется точной метрикой текущего commit; таблица выше сохраняется как frozen baseline выбранного prompt/LLM pipeline на момент независимого прогона.

## Логи и observability

Для Docker:

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

Клиент может передать собственный `X-Request-ID`; если заголовок отсутствует, приложение генерирует UUID. `X-Request-ID` также возвращается в HTTP response headers.

Ожидаемые ошибки OpenAI-compatible LLM нормализуются на границе LLM-клиента и логируются как структурированное событие. Внутренние ошибки приложения не маскируются как «LLM недоступна».

## Rate limiting

Rate limiting применяется к `/search`, `/ask` и `/articles/{number}`.

Реализация — простой in-memory limiter с отдельным sliding window для каждого клиента. Устаревшие client IDs периодически удаляются из памяти, поэтому адреса клиентов, которые больше не обращаются к сервису, не накапливаются в течение всего времени жизни процесса.

В текущей Docker-конфигурации приложение запускается одним Uvicorn worker. Состояние limiter хранится только в памяти процесса и не разделяется между несколькими workers или экземплярами приложения.

Если сервис будет запущен с несколькими workers или репликами, каждый процесс будет иметь собственный независимый счётчик. Для строгого общего лимита потребуется shared storage, например Redis, либо rate limiting на уровне reverse proxy/API gateway.

В текущем deployment reverse proxy перед приложением не используется, поэтому идентификатор клиента определяется через непосредственный адрес соединения (`request.client.host`).

Приложение намеренно не доверяет произвольному `X-Forwarded-For`: при прямом доступе клиент может подделать этот заголовок и обходить IP-based rate limit. Если сервис будет размещён за nginx, Traefik или другим reverse proxy, trusted proxy headers необходимо настраивать отдельно на уровне deployment.

## Quality checks и CI

Локальные проверки:

```bash
ruff check .
pyright
python -m pytest
git diff --check
```

Tool configuration хранится в корневом `pyproject.toml`:

- Ruff target — Python 3.14;
- Pyright — Python 3.14, `standard` mode;
- Pytest — зарегистрирован marker `integration` для тестов с реальными внешними библиотеками/локальным model inference.

GitHub Actions workflow запускается на pull request и выполняет:

1. checkout репозитория;
2. установку Python 3.14;
3. установку `requirements.txt`;
4. `ruff check .`;
5. `pyright`;
6. `python -m pytest`.

Таким образом, lint, static type checking и тесты проверяются автоматически до merge.

## Известные ограничения

- Источник Конституции не обновляется автоматически; актуальность `data/raw/constitution.txt` контролируется пользователем проекта.
- `min_score=0.833` откалиброван на dev-наборе. Frozen holdout показывает пересечение cosine distributions на более сложных negative-запросах, поэтому абсолютный threshold не является универсальным классификатором релевантности.
- Hybrid retrieval опционален и по умолчанию выключен.
- Generation groundedness не равна 100%; сложные synthesis-вопросы остаются основной зоной риска.
- Rate limiter рассчитан на текущий single-worker deployment и не является распределённым.
- Ollama не входит в Docker Compose stack и при необходимости запускается отдельно.
