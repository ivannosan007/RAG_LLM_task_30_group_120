import logging
from typing import List, Tuple, Dict
import numpy as np


from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config, vector_store):
        
        self.config = config
        self.vector_store = vector_store
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        
        self._initialize_vllm()
            
    def _initialize_vllm(self):
        
        logger.info(f"Initializing vLLM with model: {self.config.llm_model_name}")
        self.sampling_params = SamplingParams(**self.config.vllm_engine['sampling_params'])
        self.llm = LLM(model=self.config.llm_model_name)
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM initialized successfully")
        
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:

        if top_k is None:
            top_k = self.config.top_k
            
        # logger.info(f"Retrieving top {top_k} documents for query: {query}")
        results = self.vector_store.search(query, top_k)
        # logger.info(f"Retrieved {len(results)} documents")
        return results

    def generate_prompt(self, query: str, retrieved_docs: List[Tuple[str, float, str]]) -> str:
        # создаем промпт для одного запроса в модель
        doc_texts = []
        for chunk_id, score, chunk_text in retrieved_docs:
            doc_texts.append(f"Контекст (релевантность: {score:.3f}):\n{chunk_text}")
        context = "\n\n".join(doc_texts)
        prompt = f"""{self.config.prompt}\nКонтексты:\n{context}\nВопрос:{query}\nОтвет:"""
        return prompt
    
    def generate(self, prompt: str) -> str:
        
        logger.info("Generating response with vLLM")
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text
        logger.info("Response generated successfully")
        return response

    
    def prepare_prompts(self, queries: List[str]) -> tuple[List[List[Dict[str, str]]], List[List[Tuple[str, float, str]]]]:

        prompts = []
        retrieved_docs_list = []
        for query in queries:
            retrieved_docs = self.retrieve(query)
            retrieved_docs_list.append(retrieved_docs)
            user_prompt = self.generate_prompt(query, retrieved_docs)
            prompt = [
                {"role": "system", "content": "Используя следующие документы, ответь на вопрос на русском языке."},
                {"role": "user", "content": user_prompt}
            ]
            prompts.append(prompt)
        return prompts, retrieved_docs_list
    
    def batch_infer(self, queries: List[str]) -> tuple[List[str], List[List[Dict[str, str]]], List[List[Tuple[str, float, str]]]]:

        prompts, retrieved_docs_list = self.prepare_prompts(queries)
        enable_thinking = self.config.vllm_engine.get('enable_thinking', None)
        
        if enable_thinking is not None:
            texts = [
                self.tokenizer.apply_chat_template(
                    p,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for p in prompts
            ]
        else:
            texts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
        
        logger.info(f"Generating {len(texts)} responses with vLLM using chat template")
        outputs = self.llm.generate(texts, self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        logger.info("Chat template batch responses generated successfully")
        return responses, prompts, retrieved_docs_list
