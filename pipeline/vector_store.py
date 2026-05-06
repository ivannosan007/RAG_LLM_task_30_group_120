import numpy as np
import faiss
import pickle
import logging
import re
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_count = len(self.tokenized_docs)
        self.k1 = 1.5
        self.b = 0.75

        self.doc_lengths = np.array([len(doc) for doc in self.tokenized_docs], dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lengths)) if self.doc_count else 0.0

        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        for doc in self.tokenized_docs:
            counts: Dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs.append(counts)
            for token in set(doc):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.idf: Dict[str, float] = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = np.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def get_top_k(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.doc_count == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(self.doc_count, dtype=np.float32)
        for q_token in query_tokens:
            idf = self.idf.get(q_token, 0.0)
            if idf == 0.0:
                continue
            for idx, counts in enumerate(self.term_freqs):
                tf = counts.get(q_token, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (
                    1.0 - self.b + self.b * (self.doc_lengths[idx] / max(self.avg_doc_len, 1e-9))
                )
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / norm

        if top_k <= 0:
            return []

        top_indices = np.argpartition(scores, -min(top_k, self.doc_count))[-min(top_k, self.doc_count):]
        ranked = sorted(
            [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0.0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


class VectorStore:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.reranker = None
        self.index = None
        self.bm25 = None
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []

    def load_embedding_model(self):
        logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
        self.model = SentenceTransformer(self.config.embedding_model_name)
        logger.info("Embedding model loaded successfully")

    def load_reranker_model(self):
        reranker_name = getattr(self.config, "reranker_model_name", "")
        if not reranker_name:
            return
        if self.reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {reranker_name}")
            self.reranker = CrossEncoder(reranker_name)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            self.reranker = None
            logger.warning(f"Failed to load reranker model '{reranker_name}': {e}")

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return vectors / norms

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
        embeddings = self._l2_normalize(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity
        self.index.add(embeddings.astype(np.float32))

        self.chunk_ids = list(chunk_ids)
        self.chunk_texts = list(chunk_texts)
        self.bm25 = BM25Retriever(self.chunk_texts)
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
        self.bm25 = BM25Retriever(self.chunk_texts)

        logger.info(f"Index loaded from {self.config.vector_store_path}")

    def _ensure_ready(self):
        if self.index is None or self.model is None:
            self.load_embedding_model()
            self.load_index()
        if self.bm25 is None:
            self.bm25 = BM25Retriever(self.chunk_texts)

    def search_dense(self, query: str, top_k: int) -> List[Tuple[str, float, str]]:
        self._ensure_ready()

        query_embedding = self.model.encode([query])
        query_embedding = self._l2_normalize(query_embedding)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score), self.chunk_texts[idx]))
        return results

    def search_bm25(self, query: str, top_k: int) -> List[Tuple[str, float, str]]:
        self._ensure_ready()
        ranked = self.bm25.get_top_k(query, top_k)
        return [(self.chunk_ids[idx], score, self.chunk_texts[idx]) for idx, score in ranked]

    def _rrf_fuse(
        self,
        dense_results: List[Tuple[str, float, str]],
        bm25_results: List[Tuple[str, float, str]],
        top_k: int,
    ) -> List[Tuple[str, float, str]]:
        rrf_k = int(getattr(self.config, "rrf_k", 60))
        fused: Dict[str, Dict[str, object]] = {}

        for rank, (chunk_id, _, chunk_text) in enumerate(dense_results, start=1):
            fused.setdefault(chunk_id, {"score": 0.0, "text": chunk_text})
            fused[chunk_id]["score"] += 1.0 / (rrf_k + rank)

        for rank, (chunk_id, _, chunk_text) in enumerate(bm25_results, start=1):
            fused.setdefault(chunk_id, {"score": 0.0, "text": chunk_text})
            fused[chunk_id]["score"] += 1.0 / (rrf_k + rank)

        ranked = sorted(
            [(chunk_id, float(payload["score"]), str(payload["text"])) for chunk_id, payload in fused.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, str]],
        top_k: int,
    ) -> List[Tuple[str, float, str]]:
        if not candidates:
            return []

        self.load_reranker_model()
        if self.reranker is None:
            return candidates[:top_k]

        pairs = [[query, chunk_text] for _, _, chunk_text in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked = []
        for (chunk_id, _, chunk_text), score in zip(candidates, rerank_scores):
            reranked.append((chunk_id, float(score), chunk_text))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def search_staged(
        self, query: str, top_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float, str]]]:
        if top_k is None:
            top_k = int(getattr(self.config, "top_k", 5))

        dense_k = int(getattr(self.config, "dense_candidate_k", max(top_k, 20)))
        bm25_k = int(getattr(self.config, "bm25_candidate_k", max(top_k, 20)))
        rerank_k = int(getattr(self.config, "rerank_candidate_k", max(top_k, 20)))

        dense_results = self.search_dense(query, dense_k)
        bm25_results = self.search_bm25(query, bm25_k)
        hybrid_results = self._rrf_fuse(dense_results, bm25_results, rerank_k)
        reranked_results = self._rerank(query, hybrid_results, top_k)

        return {
            "dense": dense_results,
            "bm25": bm25_results,
            "hybrid": hybrid_results,
            "reranked": reranked_results,
        }

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        staged = self.search_staged(query, top_k=top_k)
        return staged["reranked"]
