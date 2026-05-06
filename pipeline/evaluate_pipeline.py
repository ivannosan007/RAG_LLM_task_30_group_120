import os
import sys
import time
import logging
import pandas as pd
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_processor import DataProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

from bert_score import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_dataset(config, sample_size=None):
    logger.info(f"Loading dataset from {config.dataset_path}")
    df = pd.read_pickle(config.dataset_path)
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42) if sample_size else df

    questions = sampled_df["question"].tolist()
    ground_truth_answers = sampled_df["answer"].tolist()
    contexts = sampled_df["context"].tolist()
    filenames = sampled_df["filename"].tolist() if "filename" in sampled_df.columns else [None] * len(questions)

    logger.info(f"Loaded {len(questions)} questions from dataset")
    return questions, ground_truth_answers, contexts, filenames


def calculate_exact_match(predicted_answers, ground_truth_answers):
    matches = 0
    for pred, gt in zip(predicted_answers, ground_truth_answers):
        if pred.strip().lower() == gt.strip().lower():
            matches += 1
    return matches / len(ground_truth_answers) if ground_truth_answers else 0


def calculate_recall_at_k(retrieved_docs_list, ground_truth_filenames, k=3):
    recalls = []
    for retrieved_docs, gt_filename in zip(retrieved_docs_list, ground_truth_filenames):
        if gt_filename is None:
            recalls.append(None)
            continue

        retrieved_chunk_ids = [doc[0] for doc in retrieved_docs]
        retrieved_filenames = [chunk_id.split("_chunk")[0] for chunk_id in retrieved_chunk_ids]
        gt_filename = gt_filename.replace(".txt", "")
        is_relevant_retrieved = gt_filename in retrieved_filenames[:k]
        recalls.append(1.0 if is_relevant_retrieved else 0.0)

    valid_recalls = [r for r in recalls if r is not None]
    return sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0


def calculate_mrr(retrieved_docs_list, ground_truth_filenames):
    reciprocal_ranks = []
    for retrieved_docs, gt_filename in zip(retrieved_docs_list, ground_truth_filenames):
        if gt_filename is None:
            reciprocal_ranks.append(None)
            continue

        retrieved_chunk_ids = [doc[0] for doc in retrieved_docs]
        retrieved_filenames = [chunk_id.split("_chunk")[0] for chunk_id in retrieved_chunk_ids]
        gt_filename = gt_filename.replace(".txt", "")
        rank = None
        for i, filename in enumerate(retrieved_filenames):
            if filename == gt_filename:
                rank = i + 1
                break

        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    valid_rr = [rr for rr in reciprocal_ranks if rr is not None]
    return sum(valid_rr) / len(valid_rr) if valid_rr else 0.0


def calculate_bert_score(predicted_answers, ground_truth_answers):
    preds = [ans.strip().lower() for ans in predicted_answers]
    gt_labels = [gt.strip().lower() for gt in ground_truth_answers]
    _, _, f1 = score(preds, gt_labels, lang="ru", verbose=False)
    return f1.mean().item()


def run_ollama_inference(
    rag_pipeline: RAGPipeline,
    questions: List[str],
) -> Tuple[
    List[str],
    List[str],
    List[List[Tuple[str, float, str]]],
    List[List[Tuple[str, float, str]]],
    List[List[Tuple[str, float, str]]],
]:
    """
    Returns:
      - predicted_answers: List[str]
      - prompts: List[str]
      - dense_docs_list: List[List[(chunk_id, score, chunk_text)]]
      - hybrid_docs_list: List[List[(chunk_id, score, chunk_text)]]
      - reranked_docs_list: List[List[(chunk_id, score, chunk_text)]]
    """
    predicted_answers: List[str] = []
    prompts: List[str] = []
    dense_docs_list: List[List[Tuple[str, float, str]]] = []
    hybrid_docs_list: List[List[Tuple[str, float, str]]] = []
    reranked_docs_list: List[List[Tuple[str, float, str]]] = []

    for i, q in enumerate(questions, start=1):
        staged = rag_pipeline.vector_store.search_staged(q, top_k=rag_pipeline.config.top_k)
        dense_docs = staged["dense"]
        hybrid_docs = staged["hybrid"]
        reranked_docs = staged["reranked"]

        dense_docs_list.append(dense_docs)
        hybrid_docs_list.append(hybrid_docs)
        reranked_docs_list.append(reranked_docs)

        prompt_text = rag_pipeline.generate_prompt(q, reranked_docs)
        prompts.append(prompt_text)

        try:
            ans = rag_pipeline.generate(prompt_text)
        except Exception as e:
            logger.error(f"Ollama generation failed on item {i}/{len(questions)}: {e}")
            ans = ""
        predicted_answers.append(ans)

        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(questions)} questions")

    return predicted_answers, prompts, dense_docs_list, hybrid_docs_list, reranked_docs_list


def main():
    config = Config()

    data_processor = DataProcessor(config)
    vector_store = VectorStore(config)

    if os.path.exists(config.vector_store_path):
        logger.info("Loading existing vector store index")
        vector_store.load_index()
    else:
        logger.info("Building new vector store index")
        texts = data_processor.process_texts()
        chunked_texts = data_processor.create_chunks(texts)
        vector_store.build_index(chunked_texts)
        vector_store.save_index()

    rag_pipeline = RAGPipeline(config, vector_store)
    questions, ground_truth_answers, contexts, filenames = load_test_dataset(config)

    logger.info(f"Processing {len(questions)} questions (Ollama mode)")

    start_time = time.time()
    predicted_answers, prompts, dense_docs_list, hybrid_docs_list, reranked_docs_list = run_ollama_inference(
        rag_pipeline, questions
    )
    end_time = time.time()

    total_latency = end_time - start_time
    avg_latency = total_latency / len(questions) if questions else 0.0

    exact_match = calculate_exact_match(predicted_answers, ground_truth_answers)

    dense_recall_at_3 = calculate_recall_at_k(dense_docs_list, filenames, k=3)
    dense_recall_at_5 = calculate_recall_at_k(dense_docs_list, filenames, k=5)
    dense_mrr = calculate_mrr(dense_docs_list, filenames)

    hybrid_recall_at_3 = calculate_recall_at_k(hybrid_docs_list, filenames, k=3)
    hybrid_recall_at_5 = calculate_recall_at_k(hybrid_docs_list, filenames, k=5)
    hybrid_mrr = calculate_mrr(hybrid_docs_list, filenames)

    reranked_recall_at_3 = calculate_recall_at_k(reranked_docs_list, filenames, k=3)
    reranked_recall_at_5 = calculate_recall_at_k(reranked_docs_list, filenames, k=5)
    reranked_mrr = calculate_mrr(reranked_docs_list, filenames)

    bert_score_f1 = calculate_bert_score(predicted_answers, ground_truth_answers)

    logger.info(f"\n{'-'*80}")
    logger.info("RETRIEVAL + GENERATION METRICS (Ollama)")
    logger.info("-" * 80)
    logger.info("Dense retrieval:")
    logger.info(f"  Recall@3: {dense_recall_at_3:.2%}")
    logger.info(f"  Recall@5: {dense_recall_at_5:.2%}")
    logger.info(f"  MRR: {dense_mrr:.4f}")

    logger.info("Hybrid retrieval (Dense + BM25 + RRF):")
    logger.info(f"  Recall@3: {hybrid_recall_at_3:.2%}")
    logger.info(f"  Recall@5: {hybrid_recall_at_5:.2%}")
    logger.info(f"  MRR: {hybrid_mrr:.4f}")

    logger.info("Reranked retrieval (final):")
    logger.info(f"  Recall@3: {reranked_recall_at_3:.2%}")
    logger.info(f"  Recall@5: {reranked_recall_at_5:.2%}")
    logger.info(f"  MRR: {reranked_mrr:.4f}")

    logger.info(
        f"Reranker uplift vs hybrid: Recall@3 {reranked_recall_at_3 - hybrid_recall_at_3:+.2%}, "
        f"MRR {reranked_mrr - hybrid_mrr:+.4f}"
    )

    logger.info(f"Exact Match Accuracy: {exact_match:.2%}")
    logger.info(f"BERTScore F1: {bert_score_f1:.4f}")
    logger.info(f"Average Latency (per query): {avg_latency:.4f} seconds")
    logger.info(f"Total Latency: {total_latency:.4f} seconds")
    logger.info(f"{'-'*80}")


if __name__ == "__main__":
    main()
