import logging
import os
from typing import List, Tuple, Dict

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaNotRunningError(ConnectionError):
    def __init__(self):
        super().__init__(
            "Ollama не запущена. Откройте отдельный терминал и выполните: ollama serve"
        )


class RAGPipeline:
    def __init__(self, config, vector_store):

        self.config = config
        self.vector_store = vector_store
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        self.backend = os.environ.get("LLM_BACKEND", "auto")
        if self.backend == "auto":
            self.backend = "vllm" if HAS_VLLM else "ollama"

        if self.backend == "vllm":
            self._initialize_vllm()
        else:
            self._setup_ollama()

    def _setup_ollama(self):
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.ollama_url = os.environ.get(
            "OLLAMA_URL", "http://localhost:11434/api/chat"
        )
        self.sampling_params = self.config.vllm_engine["sampling_params"]
        logger.info(f"Ollama backend configured (model: {self.ollama_model})")

    def is_ollama_available(self) -> bool:
        if self.backend != "ollama":
            return True
        try:
            import requests

            response = requests.get(
                self.ollama_url.replace("/api/chat", "/api/tags"),
                timeout=3,
            )
            return response.ok
        except Exception:
            return False

    def _initialize_vllm(self):
        logger.info(f"Initializing vLLM with model: {self.config.llm_model_name}")
        self.sampling_params = SamplingParams(**self.config.vllm_engine["sampling_params"])
        self.llm = LLM(model=self.config.llm_model_name)
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM initialized successfully")

    def _generate_with_ollama(self, messages: List[Dict[str, str]]) -> str:
        import requests

        if not self.is_ollama_available():
            raise OllamaNotRunningError()

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.sampling_params.get("temperature", 0.7),
                "top_p": self.sampling_params.get("top_p", 0.8),
                "num_predict": self.sampling_params.get("max_tokens", 512),
            },
        }
        response = requests.post(self.ollama_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:

        if top_k is None:
            top_k = self.config.top_k

        results = self.vector_store.search(query, top_k)
        return results

    def generate_prompt(self, query: str, retrieved_docs: List[Tuple[str, float, str]]) -> str:
        doc_texts = []
        for chunk_id, score, chunk_text in retrieved_docs:
            doc_texts.append(f"Контекст (релевантность: {score:.3f}):\n{chunk_text}")
        context = "\n\n".join(doc_texts)
        prompt = f"""{self.config.prompt}\nКонтексты:\n{context}\nВопрос:{query}\nОтвет:"""
        return prompt

    def _build_chat_messages(self, question: str, retrieved_docs: List[Tuple[str, float, str]]):
        user_prompt = self.generate_prompt(question, retrieved_docs)
        return [
            {
                "role": "system",
                "content": "Используя следующие документы, ответь на вопрос на русском языке.",
            },
            {"role": "user", "content": user_prompt},
        ]

    def query(
        self,
        question: str,
        top_k: int = None,
    ) -> tuple[str, str, List[Tuple[str, float, str]]]:
        if top_k is None:
            top_k = self.config.top_k

        retrieved_docs = self.retrieve(question, top_k=top_k)
        prompt = self.generate_prompt(question, retrieved_docs)
        messages = self._build_chat_messages(question, retrieved_docs)

        if self.backend == "ollama":
            answer = self._generate_with_ollama(messages)
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = self.llm.generate([text], self.sampling_params)[0].outputs[0].text

        return answer, prompt, retrieved_docs

    def generate(self, prompt: str) -> str:
        if self.backend == "ollama":
            logger.info("Generating response with Ollama")
            response = self._generate_with_ollama([{"role": "user", "content": prompt}])
            logger.info("Response generated successfully")
            return response

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
            prompts.append(self._build_chat_messages(query, retrieved_docs))
        return prompts, retrieved_docs_list

    def batch_infer(self, queries: List[str]) -> tuple[List[str], List[List[Dict[str, str]]], List[List[Tuple[str, float, str]]]]:

        prompts, retrieved_docs_list = self.prepare_prompts(queries)

        if self.backend == "ollama":
            logger.info(f"Generating {len(prompts)} responses with Ollama")
            responses = [self._generate_with_ollama(prompt) for prompt in prompts]
            logger.info("Ollama batch responses generated successfully")
            return responses, prompts, retrieved_docs_list

        enable_thinking = self.config.vllm_engine.get("enable_thinking", None)

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
