import os
import re
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r'\u0301', '', text)
        return cleaned
    
    def load_dataset(self) -> pd.DataFrame:
        logger.info(f"Loading dataset from {self.config.dataset_path}")
        df = pd.read_pickle(self.config.dataset_path)
        logger.info(f"Dataset loaded with {len(df)} rows")
        return df
    
    def process_texts(self) -> List[Tuple[str, str]]:
        logger.info(f"Processing texts from {self.config.files_dir}")
        
        corpus = []
        corpus_ids = []
        
        files_dir = Path(self.config.files_dir)
        if files_dir.exists() and files_dir.is_dir():
            for file_path in files_dir.glob('*'):
                try:
                    text = file_path.read_text(encoding='utf-8', errors='ignore')
                    cleaned_text = self.clean_text(text)
                    corpus.append(cleaned_text)
                    corpus_ids.append(file_path.stem)
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
                    continue
                    
            logger.info(f'Loaded {len(corpus)} files from {self.config.files_dir}')
        else:
            logger.warning(f"Directory {self.config.files_dir} does not exist")
            
        return list(zip(corpus_ids, corpus))
    
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        tokens = re.findall(r"\w+|\S", text)
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i:i+chunk_size]
            chunks.append(' '.join(chunk))
            i += chunk_size - overlap
            
        return chunks
    
    def create_chunks(self, texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

        logger.info(f"Creating chunks with size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        
        chunked_corpus = []
        chunked_ids = []
        
        for text_id, text in texts:
            chunks = self.chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
            for j, chunk in enumerate(chunks):
                chunked_corpus.append(chunk)
                chunked_ids.append(f'{text_id}_chunk{j}')
                
        logger.info(f"Created {len(chunked_corpus)} chunks")
        return list(zip(chunked_ids, chunked_corpus))
