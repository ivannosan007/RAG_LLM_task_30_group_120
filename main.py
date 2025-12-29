from fastapi import FastAPI
import os

# Отключаем multiprocessing для vLLM ДО импорта
os.environ["VLLM_USE_MULTIPROCESSING"] = "0"

from pipeline.rag_pipeline import RAGPipeline
from pipeline.vector_store import VectorStore
from pipeline.config import Config

app = FastAPI()

# Инициализация
config = Config()
vector_store = VectorStore(config)
rag = RAGPipeline(config, vector_store)

@app.get("/")
def home():
    return {"status": "RAG API ready"}

@app.post("/ask")
def ask(question: str):
    return rag.query(question=question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
