import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_processor import DataProcessor
from rag_pipeline import HAS_VLLM, RAGPipeline
from vector_store import VectorStore

# EXAMPLE_QUESTIONS = [
#     "Что может вызвать цунами?",
#     "Кто написал роман «Хижина дяди Тома»?",
#     "Кто создал Зимний дворец в Санкт-Петербурге?",
#     "Как звали отца Северуса Снейпа?",
#     "Какому музею принадлежит «Джоконда»?",
# ]


def ensure_vector_store(config: Config, vector_store: VectorStore) -> None:
    if os.path.exists(config.vector_store_path):
        vector_store.load_index()
        return

    texts = DataProcessor(config).process_texts()
    if not texts:
        raise FileNotFoundError(
            f"Не найдены документы в {config.files_dir}. "
            "Скачайте датасет ru_rag_test_dataset и положите файлы в папку files/."
        )

    chunked_texts = DataProcessor(config).create_chunks(texts)
    vector_store.build_index(chunked_texts)
    vector_store.save_index()


@st.cache_resource(show_spinner="Загрузка моделей и индекса...")
def load_pipeline():
    config = Config()
    vector_store = VectorStore(config)
    ensure_vector_store(config, vector_store)
    rag = RAGPipeline(config, vector_store)
    return config, rag


def chunk_source_id(chunk_id: str) -> str:
    return chunk_id.split("_chunk")[0]


def render_sidebar(config: Config, rag: RAGPipeline) -> tuple[int, bool, bool]:
    st.sidebar.title("Настройки")

    top_k = st.sidebar.slider(
        "Количество контекстов (top_k)",
        min_value=1,
        max_value=10,
        value=config.top_k,
    )
    show_prompt = st.sidebar.checkbox("Показать промпт для LLM", value=False)
    show_contexts = st.sidebar.checkbox("Показать найденные фрагменты", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("Модели")
    st.sidebar.markdown(f"**Embeddings:** `{config.embedding_model_name}`")

    if rag.backend == "vllm":
        st.sidebar.markdown(f"**LLM:** `{config.llm_model_name}` (vLLM)")
    else:
        st.sidebar.markdown(f"**LLM:** `{rag.ollama_model}` (Ollama)")

    st.sidebar.divider()
    st.sidebar.caption(
        "Ответ строится только на основе найденных фрагментов документов. "
        "Если информации нет — модель вернёт N/A."
    )

    return top_k, show_prompt, show_contexts


def render_contexts(retrieved_docs) -> None:
    for i, (chunk_id, score, chunk_text) in enumerate(retrieved_docs, start=1):
        source = chunk_source_id(chunk_id)
        with st.expander(f"#{i} · {source} · score {score:.3f}"):
            st.markdown(chunk_text)


def main() -> None:
    st.set_page_config(
        page_title="RAG QA — русский язык",
        page_icon="📚",
        layout="wide",
    )

    st.title("RAG — вопросы и ответы")
    st.markdown(
        "Система ищет релевантные фрагменты в коллекции документов "
        "и генерирует ответ с помощью языковой модели."
    )

    try:
        config, rag = load_pipeline()
    except Exception as exc:
        st.error(f"Не удалось загрузить пайплайн: {exc}")
        if not HAS_VLLM:
            st.info(
                "На macOS используйте Ollama: запустите `ollama serve` "
                "и при необходимости задайте `OLLAMA_MODEL=qwen2.5-coder:7b`."
            )
        st.stop()

    top_k, show_prompt, show_contexts = render_sidebar(config, rag)

    if "history" not in st.session_state:
        st.session_state.history = []

    # st.subheader("Примеры вопросов")
    # cols = st.columns(len(EXAMPLE_QUESTIONS))
    # for col, example in zip(cols, EXAMPLE_QUESTIONS):
    #     if col.button(example, use_container_width=True):
    #         st.session_state.pending_question = example

    question = st.text_input(
        "Ваш вопрос",
        value=st.session_state.pop("pending_question", ""),
        placeholder="Например: Что может вызвать цунами?",
    )

    ask = st.button("Получить ответ", type="primary", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("Введите вопрос.")
        else:
            with st.spinner("Ищем контекст и генерируем ответ..."):
                start = time.time()
                try:
                    answer, prompt, retrieved = rag.query(
                        question.strip(),
                        top_k=top_k,
                    )
                    latency = time.time() - start
                except Exception as exc:
                    st.error(f"Ошибка при генерации ответа: {exc}")
                    st.stop()

            st.session_state.history.insert(
                0,
                {
                    "question": question.strip(),
                    "answer": answer,
                    "prompt": prompt,
                    "retrieved": retrieved,
                    "latency": latency,
                },
            )

    if st.session_state.history:
        st.divider()
        st.subheader("История запросов")

        for item in st.session_state.history:
            st.markdown(f"**Вопрос:** {item['question']}")
            st.success(item["answer"])
            st.caption(f"Время ответа: {item['latency']:.1f} сек.")

            if show_contexts:
                render_contexts(item["retrieved"])

            if show_prompt:
                with st.expander("Промпт для LLM"):
                    st.code(item["prompt"])

            st.divider()


if __name__ == "__main__":
    main()
