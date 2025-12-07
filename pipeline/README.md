# RAG Pipeline for Russian Language QA

Этот проект реализует Retrieval-Augmented Generation (RAG) пайплайн для задачи ответов на вопросы на русском языке.

## Архитектура

Пайплайн состоит из нескольких компонентов:

1. **Vector Store** (`vector_store.py`) - управление векторным хранилищем для поиска релевантных документов
2. **RAG Pipeline** (`rag_pipeline.py`) - основной класс, объединяющий retrieval и generation
3. **Data Processor** (`data_processor.py`) - обработка и подготовка текстовых данных
4. **Configuration** (`config.py`, `config.yaml`) - управление конфигурацией пайплайна
5. **Evaluation** (`evaluate_pipeline.py`) - оценка качества работы пайплайна

## Требования

Для установки всех необходимых зависимостей выполните:

```bash
pip install -r requirements.txt
```

### Оценка качества

Для оценки качества работы пайплайна используется скрипт `evaluate_pipeline.py`:

```bash
python evaluate_pipeline.py
```

Скрипт автоматически загружает тестовый набор данных и вычисляет следующие метрики:
- **Exact Match Accuracy** - точное совпадение ответов
- **Recall@K** - доля случаев, когда правильный документ находится среди K первых результатов поиска
- **MRR** - Mean Reciprocal Rank
- **BERTScore F1** - семантическое сходство между предсказанными и реальными ответами

