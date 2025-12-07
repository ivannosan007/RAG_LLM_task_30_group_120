import numpy as np
import faiss
import pickle
import os
import logging
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.index = None
        
    def load_embedding_model(self):
        logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
        self.model = SentenceTransformer(self.config.embedding_model_name)
        logger.info("Embedding model loaded successfully")
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.load_embedding_model()
            
        logger.info(f"Encoding {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Encoded texts to embeddings with shape {embeddings.shape}")
        return embeddings
    
    def build_index(self, chunked_texts: List[Tuple[str, str]]):
        chunk_ids, chunk_texts = zip(*chunked_texts)
        
        embeddings = self.encode_texts(list(chunk_texts))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity
        self.index.add(embeddings.astype(np.float32))
        
        self.chunk_ids = list(chunk_ids)
        self.chunk_texts = list(chunk_texts)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
    def save_index(self):

        faiss.write_index(self.index, self.config.vector_store_path)
        chunk_data_path = self.config.vector_store_path.replace('.index', '_data.pkl')
        with open(chunk_data_path, 'wb') as f:
            pickle.dump({'ids': self.chunk_ids, 'texts': self.chunk_texts}, f)
            
        logger.info(f"Index saved to {self.config.vector_store_path}")
        
    def load_index(self):

        self.index = faiss.read_index(self.config.vector_store_path)
        chunk_data_path = self.config.vector_store_path.replace('.index', '_data.pkl')
        with open(chunk_data_path, 'rb') as f:
            data = pickle.load(f)
            self.chunk_ids = data['ids']
            self.chunk_texts = data['texts']
            
        logger.info(f"Index loaded from {self.config.vector_store_path}")
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:

        if self.index is None or self.model is None:
            self.load_embedding_model()
            self.load_index()
            
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                chunk_text = self.chunk_texts[idx]
                results.append((chunk_id, float(score), chunk_text))
                
        return results
