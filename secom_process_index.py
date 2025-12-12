#!/usr/bin/env python3
"""
SeCom Indexing Script
Process conversational data and build SeCom memory banks with segmentation and compression.
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

# Add SeCom to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def format_dialogs_for_secom(dialogs):
    """
    Format dialogs into SeCom's expected format: List[List[str]]
    Group messages by session (datetime) - concatenate all messages with the same timestamp.
    
    Args:
        dialogs: List of dialog sessions with messages
        
    Returns:
        List of sessions, where each session is a list of concatenated message strings
    """
    sessions = []
    
    for session in dialogs:
        timestamp = session.get('datetime', '')
        messages = session.get('messages', [])
        
        if not messages:
            continue
        
        # Concatenate all messages in this session (same datetime)
        session_text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # Format with role prefix
            if role in ['user', 'User', 'Caroline', 'Melanie']:
                session_text_parts.append(f"[human] {content}")
            else:
                session_text_parts.append(f"[bot] {content}")
        
        # Join all messages in this session into one string
        # Then wrap in a list because SeCom expects List[List[str]]
        # where each session can have multiple "turns" but we group everything from same datetime
        if session_text_parts:
            # Combine all message parts into single session string
            session_combined = " ".join(session_text_parts)
            sessions.append([session_combined])  # Wrap in list for SeCom format
    
    return sessions


def process_indexing(
    input_file: str,
    base_output_dir: str,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    llm_model: str = "gpt-4o-mini",
    api_key: str = "dummy",
    base_url: str = None,
    granularity: str = "segment",
    compress_rate: float = 0.9,
    disable_thinking: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
):
    """
    Index conversations into SeCom memory banks with segmentation and compression.
    
    Args:
        input_file: Path to LoCoMo/LongMemEval JSON dataset
        base_output_dir: Directory to save memory banks
        embedding_model: HuggingFace embedding model
        llm_model: LLM model for segmentation
        api_key: API key for LLM
        base_url: Base URL for LLM API
        granularity: Memory granularity (segment/session/turn)
        compress_rate: Compression rate for denoising (0.0-1.0)
        disable_thinking: Not used for SeCom but kept for compatibility
        num_shards: Total number of shards for parallel processing
        shard_id: Current shard ID (0-indexed)
    """
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} conversations")
    print(f"Processing shard {shard_id + 1}/{num_shards}")
    print(f"Memory granularity: {granularity}")
    print(f"Compression rate: {compress_rate}")
    print(f"Embedding model: {embedding_model}")
    print(f"LLM model: {llm_model}")
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Store metadata about indexed conversations
    index_metadata = []
    
    # Configure OpenAI API for segmentation (using self-hosted LLM)
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Determine config path based on embedding model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if "mpnet" in embedding_model.lower():
        config_path = os.path.join(script_dir, "secom/configs/mpnet.yaml")
    elif "contriever" in embedding_model.lower():
        config_path = os.path.join(script_dir, "secom/configs/contriever.yaml")
    else:
        # Default to mpnet config
        config_path = os.path.join(script_dir, "secom/configs/mpnet.yaml")
        print(f"⚠️  Unknown embedding model, using default config: {config_path}")
    
    # Process each conversation
    for conv_idx, conversation in enumerate(tqdm(dataset, desc=f"Indexing (shard {shard_id}/{num_shards})")):
        # Sharding check
        if conv_idx % num_shards != shard_id:
            continue
            
        conv_id = conversation.get("conv_id", f"conv_{conv_idx}")
        dialogs = conversation.get('dialogs', [])
        
        if not dialogs:
            print(f"\n⚠️  Conversation {conv_id}: No dialogs found, skipping")
            continue
        
        # Create directory for this conversation
        conv_dir = os.path.join(base_output_dir, conv_id)
        os.makedirs(conv_dir, exist_ok=True)
        
        print(f"\nProcessing conversation {conv_idx + 1}/{len(dataset)}: {conv_id}")
        print(f"  Sessions: {len(dialogs)}")
        
        try:
            # Import SeCom here to avoid early import errors
            from secom import SeCom
            
            # Format dialogs for SeCom
            conversation_history = format_dialogs_for_secom(dialogs)
            total_turns = sum(len(session) for session in conversation_history)
            print(f"  Total turns: {total_turns}")
            
            # Initialize SeCom with configuration
            memory_manager = SeCom(
                granularity=granularity,
                config_path=config_path
            )
            
            # Override segmentation model to use command-line LLM model
            # SeCom config files hardcode gpt-4o-mini, but we want to use our self-hosted model
            if memory_manager.segmentor:
                from secom.utils import OpenAILLM
                memory_manager.segment_model = llm_model
                memory_manager.segmentor = OpenAILLM(llm_model)
            
            # Build memory bank with segmentation and compression
            print(f"  Building memory with {granularity} granularity...")
            print(f"  Using segmentation model: {llm_model}")
            memory_manager.build_memory(
                conversation_history=conversation_history,
                compress_rate=compress_rate
            )
            
            # Initialize retriever to persist vector store
            # Using Chroma for persistent storage
            print(f"  Initializing retriever with Chroma...")
            
            import chromadb
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain_core.documents import Document
            
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cuda"}
            )
            
            # Prepare documents for Chroma - need to handle metadata properly
            # SeCom stores original content in metadata["content"] as a list
            # Chroma doesn't accept lists, so we need to convert to string
            prepared_docs = []
            for doc in memory_manager.memory_bank:
                # doc.page_content is the compressed version
                # doc.metadata["content"] is the original (list of strings)
                original_content = doc.metadata.get("content", [])
                
                # Convert list to string with proper formatting
                if isinstance(original_content, list):
                    original_content_str = "\n".join(original_content)
                else:
                    original_content_str = str(original_content)
                
                # Create new document with metadata that Chroma can handle
                new_doc = Document(
                    page_content=doc.page_content,  # compressed content for retrieval
                    metadata={
                        "original_content": original_content_str,  # Store as string, not list
                        "idx": doc.metadata.get("idx", 0),
                        "granularity": granularity,
                    }
                )
                prepared_docs.append(new_doc)
            
            # Create persistent Chroma vector store with prepared docs
            vector_store = Chroma.from_documents(
                documents=prepared_docs,
                embedding=embeddings,
                persist_directory=conv_dir,
                collection_name="memories"
            )
            
            print(f"  ✅ Persisted {len(memory_manager.memory_bank)} memory units to {conv_dir}")
            
            # Save metadata
            metadata = {
                "conv_id": conv_id,
                "num_sessions": len(dialogs),
                "num_turns": total_turns,
                "num_memory_units": len(memory_manager.memory_bank),
                "persist_directory": conv_dir,
                "embedding_model": embedding_model,
                "llm_model": llm_model,
                "granularity": granularity,
                "compress_rate": compress_rate,
                "config_path": config_path,
            }
            
            # Save metadata to JSON
            metadata_file = os.path.join(conv_dir, "memory_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            index_metadata.append(metadata)
            
            print(f"  ✅ Successfully indexed conversation {conv_id}")
            print(f"     Memory units created: {len(memory_manager.memory_bank)}")
            print(f"     Metadata saved to: {metadata_file}")
            
        except Exception as e:
            print(f"  ❌ Error processing conversation {conv_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save index metadata
    metadata_file = os.path.join(base_output_dir, f"index_metadata_{shard_id}.json")
    with open(metadata_file, 'w') as f:
        json.dump(index_metadata, f, indent=2)
    
    print(f"\n✅ Indexing completed!")
    print(f"   Processed {len(index_metadata)} conversations")
    print(f"   Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Index conversations into SeCom memory banks")
    parser.add_argument("input_file", type=str, help="Input JSON file with conversations")
    parser.add_argument("output_dir", type=str, help="Output directory for memory banks")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="HuggingFace embedding model")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                        help="LLM model for segmentation")
    parser.add_argument("--api_key", type=str, default="dummy",
                        help="API key for LLM")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for LLM API")
    parser.add_argument("--granularity", type=str, default="segment",
                        choices=["segment", "session", "turn"],
                        help="Memory granularity")
    parser.add_argument("--compress_rate", type=float, default=0.75,
                        help="Compression rate for denoising (0.0-1.0)")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Disable thinking (kept for compatibility)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Current shard ID (0-indexed)")
    
    args = parser.parse_args()
    
    process_indexing(
        input_file=args.input_file,
        base_output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        granularity=args.granularity,
        compress_rate=args.compress_rate,
        disable_thinking=args.disable_thinking,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )


if __name__ == "__main__":
    main()
