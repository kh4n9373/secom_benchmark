#!/usr/bin/env python3
"""
SeCom Full Benchmark Pipeline
Orchestrates the complete benchmark: Index -> Retrieve -> Evaluate (Retrieval + Generation)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any


def run_command(cmd: list, description: str, log_file: str = None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    if log_file:
        print(f"Logging to: {log_file}")
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ {description} FAILED (exit code {result.returncode}, took {duration:.2f}s)")
        sys.exit(1)
    else:
        print(f"✅ {description} completed successfully ({duration:.2f}s)")
    
    return duration


def print_results_summary(
    retrieval_results_path: str,
    generation_results_path: str,
    total_time: float
):
    """Print a nice summary of all results."""
    print("\n" + "="*60)
    print("📊 SECOM BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    # Load retrieval results
    if os.path.exists(retrieval_results_path):
        with open(retrieval_results_path, 'r') as f:
            ret_data = json.load(f)
        
        print("\n--- RETRIEVAL EVALUATION ---")
        macro_avgs = ret_data.get("macro_avgs", {})
        micro_avgs = ret_data.get("micro_avgs", {})
        
        for k in sorted(macro_avgs.keys()):
            ma = macro_avgs[k]
            mi = micro_avgs[k]
            print(f"\n@ k={k}")
            print(f"  Macro: P={ma['precision']:.4f}  R={ma['recall']:.4f}  F1={ma['f1']:.4f}  nDCG={ma['ndcg']:.4f}")
            print(f"  Micro: P={mi['precision']:.4f}  R={mi['recall']:.4f}  F1={mi['f1']:.4f}  nDCG={mi['ndcg']:.4f}")
        
        # Category breakdown
        category_avgs = ret_data.get("category_avgs", {})
        if category_avgs:
            print("\n  By Category:")
            for cat in sorted(category_avgs.keys()):
                cat_data = category_avgs[cat]
                k = "10"  # Show k=10 for categories
                if k in cat_data.get("macro_avgs", {}):
                    ma = cat_data["macro_avgs"][k]
                    print(f"    Cat {cat}: P={ma['precision']:.4f}  R={ma['recall']:.4f}  F1={ma['f1']:.4f}  nDCG={ma['ndcg']:.4f}")
    
    # Load generation results
    if os.path.exists(generation_results_path):
        with open(generation_results_path, 'r') as f:
            gen_data = json.load(f)
        
        print("\n--- GENERATION EVALUATION ---")
        overall = gen_data.get("overall", {})
        print(f"\nOverall (n={overall.get('count', 0)})")
        print(f"  F1:           {overall.get('f1', 0):.4f}")
        print(f"  BLEU:         {overall.get('bleu', 0):.4f}")
        print(f"  ROUGE-1:      {overall.get('rouge1', 0):.4f}")
        print(f"  ROUGE-2:      {overall.get('rouge2', 0):.4f}")
        print(f"  ROUGE-L:      {overall.get('rougeL', 0):.4f}")
        print(f"  BERTScore-F1: {overall.get('bertscore_f1', 0):.4f}")
        
        # Category breakdown
        by_category = gen_data.get("by_category", {})
        if by_category:
            print("\n  By Category:")
            for cat in sorted(by_category.keys()):
                metrics = by_category[cat]
                print(f"    Cat {cat} (n={metrics.get('count', 0)}): F1={metrics.get('f1', 0):.4f}  BLEU={metrics.get('bleu', 0):.4f}  BERTScore={metrics.get('bertscore_f1', 0):.4f}")
    
    print("\n" + "="*60)
    print(f"⏱️  Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="SeCom Full Benchmark Pipeline: Index -> Retrieve -> Evaluate"
    )
    
    # Required args
    parser.add_argument("dataset_file", help="Path to input dataset JSON")
    parser.add_argument("memory_dir", help="Path to store SeCom memory banks")
    parser.add_argument("output_dir", help="Path to store all output results")
    
    # Parallel processing
    parser.add_argument("--max_workers", type=int, default=2, help="Number of parallel indexing workers")
    
    # LLM config
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="LLM model for segmentation")
    parser.add_argument("--api_key", default="dummy", help="API key for LLM")
    parser.add_argument("--base_url", default=None, help="LLM API base URL")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking (kept for compatibility)")
    
    # Embedding config
    parser.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2", 
                        help="Embedding model for retrieval")
    
    # SeCom specific
    parser.add_argument("--granularity", default="segment", choices=["segment", "session", "turn"],
                        help="Memory granularity")
    parser.add_argument("--compress_rate", type=float, default=0.75,
                        help="Compression rate for denoising")
    
    # Retrieval config
    parser.add_argument("--top_k", type=int, default=100, help="Number of memories to retrieve")
    parser.add_argument("--eval_ks", default="3,5,10", help="K values for retrieval evaluation")
    
    # Generation config
    parser.add_argument("--context_k", type=int, default=5, help="Number of chunks to use as context")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    dataset_file = os.path.abspath(args.dataset_file)
    memory_dir = os.path.abspath(args.memory_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output paths
    retrieval_output = os.path.join(output_dir, f"retrieval_results_{timestamp}.json")
    retrieval_eval_output = os.path.join(output_dir, f"retrieval_eval_{timestamp}.json")
    generation_eval_output = os.path.join(output_dir, f"generation_eval_{timestamp}.json")
    pipeline_metadata_output = os.path.join(output_dir, f"pipeline_metadata_{timestamp}.json")
    
    # Find script paths (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_script = os.path.join(script_dir, "secom_process_index.py")
    retrieve_script = os.path.join(script_dir, "secom_process_retrieve.py")
    retrieval_eval_script = os.path.join(os.path.dirname(script_dir), "retrieval_evaluator.py")
    generation_eval_script = os.path.join(os.path.dirname(script_dir), "generation_evaluator.py")
    
    # Start timer
    pipeline_start = time.time()
    timings = {}
    
    print("\n" + "="*60)
    print("🚀 SECOM BENCHMARK PIPELINE")
    print("="*60)
    print(f"Dataset: {dataset_file}")
    print(f"Memory Dir: {memory_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Workers: {args.max_workers}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Granularity: {args.granularity}")
    print(f"Compression Rate: {args.compress_rate}")
    print("="*60)
    
    # ============================================================
    # STEP 1: INDEX
    # ============================================================
    print("\n📝 STEP 1: Indexing with SeCom...")
    
    if args.max_workers > 1:
        # Parallel indexing
        print(f"Running parallel indexing with {args.max_workers} workers...")
        index_processes = []
        
        for worker_id in range(args.max_workers):
            index_cmd = [
                "python3", index_script,
                dataset_file,
                memory_dir,
                "--embedding_model", args.embedding_model,
                "--llm_model", args.llm_model,
                "--api_key", args.api_key,
                "--granularity", args.granularity,
                "--compress_rate", str(args.compress_rate),
                "--num_shards", str(args.max_workers),
                "--shard_id", str(worker_id),
            ]
            
            if args.base_url:
                index_cmd.extend(["--base_url", args.base_url])
            
            if args.disable_thinking:
                index_cmd.append("--disable_thinking")
            
            log_file = os.path.join(script_dir, "worker_logs", f"worker_{worker_id}.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            print(f"  Starting worker {worker_id}...")
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(index_cmd, stdout=f, stderr=subprocess.STDOUT)
                index_processes.append((proc, worker_id, log_file))
        
        # Wait for all workers
        print(f"\n⏳ Waiting for {args.max_workers} workers to complete...")
        all_success = True
        for proc, worker_id, log_file in index_processes:
            proc.wait()
            if proc.returncode != 0:
                print(f"❌ Worker {worker_id} failed! Check log: {log_file}")
                all_success = False
            else:
                print(f"✅ Worker {worker_id} completed")
        
        if not all_success:
            sys.exit(1)
        
        timings['indexing'] = time.time() - pipeline_start
    else:
        # Single worker
        index_cmd = [
            "python3", index_script,
            dataset_file,
            memory_dir,
            "--embedding_model", args.embedding_model,
            "--llm_model", args.llm_model,
            "--api_key", args.api_key,
            "--granularity", args.granularity,
            "--compress_rate", str(args.compress_rate),
        ]
        
        if args.base_url:
            index_cmd.extend(["--base_url", args.base_url])
        
        if args.disable_thinking:
            index_cmd.append("--disable_thinking")
        
        timings['indexing'] = run_command(index_cmd, "Indexing")
    
    # ============================================================
    # STEP 2: RETRIEVE
    # ============================================================
    print("\n🔍 STEP 2: Retrieving memories...")
    
    retrieve_cmd = [
        "python3", retrieve_script,
        dataset_file,
        memory_dir,
        retrieval_output,
        "--embedding_model", args.embedding_model,
        "--top_k", str(args.top_k),
    ]
    
    timings['retrieval'] = run_command(retrieve_cmd, "Retrieval")
    
    # ============================================================
    # STEP 3: RETRIEVAL EVALUATION
    # ============================================================
    print("\n📊 STEP 3: Evaluating retrieval quality...")
    
    retrieval_eval_cmd = [
        "python3", retrieval_eval_script,
        "--input", retrieval_output,
        "--out", retrieval_eval_output,
        "--ks", args.eval_ks,
    ]
    
    timings['retrieval_eval'] = run_command(retrieval_eval_cmd, "Retrieval Evaluation")
    
    # ============================================================
    # STEP 4: GENERATION EVALUATION
    # ============================================================
    print("\n🤖 STEP 4: Evaluating generation quality...")
    
    generation_eval_cmd = [
        "python3", generation_eval_script,
        retrieval_output,
        "--ground-truth", dataset_file,
        "--output", generation_eval_output,
        "--llm_model", args.llm_model,
        "--api_key", args.api_key,
        "--context-k", str(args.context_k),
    ]
    
    if args.base_url:
        generation_eval_cmd.extend(["--base_url", args.base_url])
    
    if args.disable_thinking:
        generation_eval_cmd.append("--disable_thinking")
    
    if args.limit:
        generation_eval_cmd.extend(["--limit", str(args.limit)])
    
    timings['generation_eval'] = run_command(generation_eval_cmd, "Generation Evaluation")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - pipeline_start
    timings['total'] = total_time
    
    # Save pipeline metadata
    metadata = {
        "dataset": dataset_file,
        "memory_dir": memory_dir,
        "output_dir": output_dir,
        "timestamp": timestamp,
        "config": {
            "max_workers": args.max_workers,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "granularity": args.granularity,
            "compress_rate": args.compress_rate,
            "top_k": args.top_k,
            "context_k": args.context_k,
            "eval_ks": args.eval_ks,
        },
        "outputs": {
            "retrieval_results": retrieval_output,
            "retrieval_eval": retrieval_eval_output,
            "generation_eval": generation_eval_output,
        },
        "timings": timings,
    }
    
    with open(pipeline_metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print_results_summary(retrieval_eval_output, generation_eval_output, total_time)
    
    print(f"\n📄 Pipeline metadata saved to: {pipeline_metadata_output}")
    print(f"📄 Retrieval results: {retrieval_output}")
    print(f"📄 Retrieval eval: {retrieval_eval_output}")
    print(f"📄 Generation eval: {generation_eval_output}")
    
    print("\n✅ SeCom benchmark pipeline completed successfully!")


if __name__ == "__main__":
    main()
