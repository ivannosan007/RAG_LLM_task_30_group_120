from pipeline.config import Config
from pipeline.data_processor import DataProcessor
from pipeline.vector_store import VectorStore

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config)
    vector_store = VectorStore(config)

    # 1. Загрузка и нарезка данных
    texts_with_ids = processor.process_texts() 
    chunks_with_ids = processor.create_chunks(texts_with_ids) 

    # 2. Создание индекса FAISS и файла .pkl
    vector_store.build_index(chunks_with_ids) 
    vector_store.save_index() 
    
    print("✅ Индексация завершена. Файлы .index и .pkl созданы.")