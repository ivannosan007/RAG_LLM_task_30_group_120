# RAG Pipeline для ответов на вопросы по документам

RAG (Retrieval-Augmented Generation) пайплайн для генерации ответов на вопросы на основе извлечённых документов.

## Компоненты пайплайна

1. **Dense Retrieval** — поиск с помощью векторных эмбеддингов (SentenceTransformer)
2. **BM25** — классический текстовый поиск
3. **Hybrid Retrieval** — комбинация Dense + BM25 с использованием Reciprocal Rank Fusion (RRF)
4. **Reranker** — переупорядочивание результатов с помощью CrossEncoder модели

## Модели

- **Embedding модель**: `intfloat/multilingual-e5-small`
- **LLM**: `qwen2.5:0.5b-instruct` (через Ollama)
- **Reranker**: `BAAI/bge-reranker-v2-m3` или `Qwen/Qwen3-Reranker-0.6B`

## Результаты экспериментов

### Сравнение reranker моделей

#### BAAI/bge-reranker-v2-m3(0.6B parameters)

```
Dense retrieval:
  Recall@3: 97.29%
  Recall@5: 97.62%
  MRR: 0.9550

Hybrid retrieval (Dense + BM25 + RRF):
  Recall@3: 96.64%
  Recall@5: 97.83%
  MRR: 0.9416

Reranked retrieval (final):
  Recall@3: 97.40%
  Recall@5: 97.40%
  MRR: 0.9632

Reranker uplift vs hybrid: Recall@3 +0.76%, MRR +0.0215

Generation metrics:
  Exact Match Accuracy: 3.25%
  BERTScore F1: 0.6595
  Average Latency: 0.9482 sec/query
  Total Latency: 875.22 sec
```

#### Qwen/Qwen3-Reranker-0.6B

```
Dense retrieval:
  Recall@3: 97.29%
  Recall@5: 97.62%
  MRR: 0.9550

Hybrid retrieval (Dense + BM25 + RRF):
  Recall@3: 96.64%
  Recall@5: 97.83%
  MRR: 0.9416

Reranked retrieval (final):
  Recall@3: 94.58%
  Recall@5: 94.58%
  MRR: 0.8324

Reranker uplift vs hybrid: Recall@3 -2.06%, MRR -0.1092

Generation metrics:
  Exact Match Accuracy: 1.95%
  BERTScore F1: 0.6522
  Average Latency: 0.7274 sec/query
  Total Latency: 671.35 sec
```