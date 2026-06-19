import os
import sys
import time
import logging
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_processor import DataProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

from bert_score import score
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_dataset(config, sample_size=None):
    logger.info(f"Loading dataset from {config.dataset_path}")
    df = pd.read_pickle(config.dataset_path)
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42) if sample_size else df
    
    questions = sampled_df['question'].tolist()
    ground_truth_answers = sampled_df['answer'].tolist()
    contexts = sampled_df['context'].tolist()
    filenames = sampled_df['filename'].tolist() if 'filename' in sampled_df.columns else [None] * len(questions)
    
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
        retrieved_filenames = [chunk_id.split('_chunk')[0] for chunk_id in retrieved_chunk_ids]
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
        retrieved_filenames = [chunk_id.split('_chunk')[0] for chunk_id in retrieved_chunk_ids]
        gt_filename = gt_filename.replace(".txt", "")
        rank = None
        for i, filename in enumerate(retrieved_filenames):
            if filename == gt_filename:
                rank = i + 1
                break
                
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    valid_rr = [rr for rr in reciprocal_ranks if rr is not None]
    return sum(valid_rr) / len(valid_rr) if valid_rr else 0.0

def calculate_bert_score(predicted_answers, ground_truth_answers):
    preds = [ans.strip().lower() for ans in predicted_answers]
    gt_labels = [gt.strip().lower() for gt in ground_truth_answers]
    
    P, R, F1 = score(preds, gt_labels, lang='ru', verbose=False)
    # logger.info("BERTScore F1 detailed comparison:")
    for i, (pred, gt, f1) in enumerate(zip(predicted_answers, ground_truth_answers, F1)):
        if pred == "N/A":
            F1[i] = 0.0

    return F1.mean().item()


def main():
    config = Config()
    
    mlflow.set_tracking_uri(config.mlflow['tracking_uri'])
    mlflow.set_experiment(config.mlflow['experiment_name'])
    
    data_processor = DataProcessor(config)
    vector_store = VectorStore(config)
    
    # check if index exists
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
    
    logger.info(f"Processing {len(questions)} questions")

    start_time = time.time()
    predicted_answers, prompts, retrieved_docs_list = rag_pipeline.batch_infer(questions)
    end_time = time.time()
    total_latency = end_time - start_time
    avg_latency = total_latency / len(questions)
    
    # calculate metrics
    exact_match = calculate_exact_match(predicted_answers, ground_truth_answers)
    recall_at_3 = calculate_recall_at_k(retrieved_docs_list, filenames, k=3)
    recall_at_5 = calculate_recall_at_k(retrieved_docs_list, filenames, k=5)
    mrr = calculate_mrr(retrieved_docs_list, filenames)
    bert_score_value = calculate_bert_score(predicted_answers, ground_truth_answers)
    
    logger.info(f"\n{'-'*80}")
    logger.info("EVALUATION METRICS")
    logger.info('-'*80)
    logger.info(f"Exact Match Accuracy: {exact_match:.2%}")
    logger.info(f"Recall@3: {recall_at_3:.2%}")
    logger.info(f"Recall@5: {recall_at_5:.2%}")
    logger.info(f"MRR: {mrr:.4f}")
    logger.info(f"BERTScore F1: {bert_score_value:.4f}")
    logger.info(f"Average Latency (per query): {avg_latency:.4f} seconds")
    logger.info(f"Total Latency: {total_latency:.4f} seconds")
    logger.info(f"{'-'*80}")
    
    with mlflow.start_run():
        # log parameters
        mlflow.log_param("embedding_model_name", config.embedding_model_name)
        mlflow.log_param("llm_model_name", config.llm_model_name)
        mlflow.log_param("top_k", config.top_k)
        mlflow.log_param("chunk_size", config.chunk_size)
        mlflow.log_param("chunk_overlap", config.chunk_overlap)
        mlflow.log_param("dataset_path", config.dataset_path)
        mlflow.log_param("num_questions", len(questions))
        
        # log vLLM sampling parameters
        sampling_params = config.vllm_engine['sampling_params']
        mlflow.log_param("vllm_max_tokens", sampling_params['max_tokens'])
        mlflow.log_param("vllm_temperature", sampling_params['temperature'])
        mlflow.log_param("vllm_top_p", sampling_params['top_p'])
        mlflow.log_param("vllm_top_k", sampling_params['top_k'])
        
        # log metrics
        mlflow.log_metric("exact_match_accuracy", exact_match)
        mlflow.log_metric("recall_at_3", recall_at_3)
        mlflow.log_metric("recall_at_5", recall_at_5)
        mlflow.log_metric("mrr", mrr)
        mlflow.log_metric("bert_score_f1", bert_score_value)
        mlflow.log_metric("avg_latency_sec", avg_latency)
        mlflow.log_metric("total_latency_sec", total_latency)
        
        os.makedirs(config.mlflow['artifact_location'], exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'question': questions,
            'predicted_answer': predicted_answers,
            'ground_truth_answer': ground_truth_answers,
            'filename': filenames
        })
        predictions_path = os.path.join(config.mlflow['artifact_location'], 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')
        mlflow.log_artifact(predictions_path, "predictions")
        
        results = {
            'exact_match_accuracy': exact_match,
            'recall_at_3': recall_at_3,
            'recall_at_5': recall_at_5,
            'mrr': mrr,
            'bert_score_f1': bert_score_value,
            'avg_latency_sec': avg_latency,
            'total_latency_sec': total_latency,
            'num_questions': len(questions)
        }
        results_path = os.path.join(config.mlflow['artifact_location'], 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(results_path, "results")
        
        logger.info(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
        

if __name__ == "__main__":
    main()
