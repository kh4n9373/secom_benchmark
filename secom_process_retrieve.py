#!/usr/bin/env python3
"""
SeCom Retrieval Script
Retrieve relevant memories for questions using SeCom's compression-based retrieval.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm

# Add SeCom to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def process_retrieval(
    input_file: str,
    memory_dir: str,
    output_file: str,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    top_k: int = 100,
):
    """
    Retrieve memories for questions using SeCom's persisted memory banks.
    
    Args:
        input_file: Path to LoCoMo/LongMemEval JSON dataset with questions
        memory_dir: Directory containing indexed memory banks
        output_file: Output file for retrieval results
        embedding_model: HuggingFace embedding model (must match indexing)
        top_k: Number of memories to retrieve per question
    """
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} conversations")
    print(f"Retrieving top-{top_k} memories per question")
    print(f"Embedding model: {embedding_model}")
    
    results = []
    total_questions = 0
    processed_questions = 0
    
    # Import required libraries
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Process each conversation
    for conv_idx, conversation in enumerate(tqdm(dataset, desc="Retrieving")):
        conv_id = conversation.get("conv_id", f"conv_{conv_idx}")
        questions = conversation.get("qas", [])
        
        if not questions:
            continue
        
        total_questions += len(questions)
        
        # Load memory metadata for this conversation
        conv_dir = os.path.join(memory_dir, conv_id)
        metadata_file = os.path.join(conv_dir, "memory_metadata.json")
        
        if not os.path.exists(metadata_file):
            print(f"\n⚠️  Memory metadata not found for {conv_id}, skipping")
            continue
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load persisted Chroma vector store
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cuda"}
            )
            
            vector_store = Chroma(
                persist_directory=conv_dir,
                embedding_function=embeddings,
                collection_name="memories"
            )
            
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            
            # Process each question
            for q_idx, question in enumerate(questions):
                question_id = question.get("question_id", f"{conv_id}_q{q_idx}")
                question_text = question.get("question", "")
                answer = question.get("answer", "")
                evidences = question.get("evidences", [])
                category = question.get("category")
                
                if not question_text:
                    continue
                
                try:
                    # Retrieve using SeCom retriever
                    retrieved_docs = retriever.invoke(question_text)
                    
                    # Format retrieved chunks
                    retrieved_chunks = []
                    for i, doc in enumerate(retrieved_docs):
                        # Extract original content from metadata
                        # We stored it as "original_content" string in indexing
                        original_content = doc.metadata.get("original_content", "")
                        
                        # If no original content, fall back to page_content (compressed)
                        if not original_content:
                            original_content = doc.page_content
                        
                        # Use page_content (compressed) for scoring but return original for comparison
                        retrieved_chunks.append({
                            "id": f"{conv_id}_mem_{i}",
                            "content": original_content,  # Original uncompressed content for evaluation
                            "score": 1.0 - (i / len(retrieved_docs)),  # Simple ranking score
                            "compressed_content": doc.page_content,  # Compressed version (optional)
                            "metadata": {
                                "idx": doc.metadata.get("idx", i),
                                "granularity": metadata.get("granularity", "segment"),
                            }
                        })
                    
                    # Add to results
                    results.append({
                        "question_id": question_id,
                        "question": question_text,
                        "answer": answer,
                        "chunks": retrieved_chunks,
                        "evidences": evidences,
                        "category": category,
                        "conv_id": conv_id
                    })
                    
                    processed_questions += 1
                    
                except Exception as e:
                    print(f"\n  Error retrieving for question {question_id}: {e}")
                    continue
        
        except Exception as e:
            print(f"\n❌ Error loading memory system for {conv_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save retrieval results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Retrieval completed!")
    print(f"   Total questions: {total_questions}")
    print(f"   Processed questions: {processed_questions}")
    print(f"   Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieve memories for questions using SeCom")
    parser.add_argument("input_file", type=str, help="Input JSON file with conversations and questions")
    parser.add_argument("memory_dir", type=str, help="Directory containing indexed memory banks")
    parser.add_argument("output_file", type=str, help="Output file for retrieval results")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="HuggingFace embedding model (must match indexing)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of memories to retrieve per question")
    
    args = parser.parse_args()
    
    process_retrieval(
        input_file=args.input_file,
        memory_dir=args.memory_dir,
        output_file=args.output_file,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
