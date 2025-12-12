#!/bin/bash
# SeCom Benchmark Setup Script
# Install dependencies and download datasets for SeCom benchmark

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🔧 SeCom Benchmark Setup"
echo "============================================================"
echo ""

# 1. Check Python version
echo "▶ Checking Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "   Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✅ Python version OK"
echo ""

# 2. Set TMPDIR to avoid disk space issues
echo "▶ Setting TMPDIR to avoid disk space issues..."
export TMPDIR="$SCRIPT_DIR/.tmp"
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"
echo "✅ TMPDIR set to $TMPDIR"
echo ""

# 3. Install SeCom package
echo "▶ Installing SeCom package..."
if [ -f "setup.py" ]; then
    pip install -e . --quiet
    echo "✅ SeCom package installed"
else
    echo "❌ setup.py not found"
    exit 1
fi
echo ""

# 4. Install additional dependencies
echo "▶ Installing additional dependencies..."
echo "   This may take a few minutes..."
echo ""

# Install llmlingua for compression
pip install llmlingua --quiet
echo "✅ llmlingua installed"

# Install langchain and related packages
pip install langchain langchain-community --quiet
echo "✅ langchain installed"

# Install vector store and embeddings
pip install chromadb sentence-transformers faiss-cpu --quiet
echo "✅ chromadb, sentence-transformers, faiss installed"

# Install evaluation packages
pip install rouge-score sacrebleu bert-score nltk --quiet
echo "✅ evaluation packages installed"

# Install other utilities
pip install python-dotenv tqdm omegaconf tiktoken --quiet
echo "✅ utility packages installed"

echo ""

# 5. Download NLTK data
echo "▶ Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true
echo "✅ NLTK data downloaded"
echo ""

# 6. Check/create data symlink or download dataset
echo "▶ Checking dataset..."
if [ -L "data" ] || [ -d "data" ]; then
    echo "✅ Data directory exists"
    
    # Check for specific datasets
    LOCOMO_FILE="data/locomo/processed_data/locomo_processed_data.json"
    LONGMEMEVAL_FILE="data/locomo/processed_data/longmemeval_processed_data.json"
    
    if [ -f "$LOCOMO_FILE" ]; then
        echo "✅ LoCoMo dataset found"
    else
        echo "⚠️  LoCoMo dataset not found at $LOCOMO_FILE"
    fi
    
    if [ -f "$LONGMEMEVAL_FILE" ]; then
        echo "✅ LongMemEval dataset found"
    else
        echo "⚠️  LongMemEval dataset not found at $LONGMEMEVAL_FILE"
    fi
else
    echo "📥 Downloading dataset from HuggingFace..."
    mkdir -p data
    
    python3 <<'EOF'
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="KhangPTT373/locomo",
        local_dir="data/locomo",
        repo_type="dataset"
    )
    print("✅ Dataset downloaded successfully!")
except Exception as e:
    print(f"❌ Failed to download dataset: {e}")
    print("   You can manually create a symlink to shared data:")
    print("   ln -s ../mem0/data data")
    exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi
echo ""

# 7. Create necessary directories
echo "▶ Creating directories..."
mkdir -p worker_logs
mkdir -p benchmark_results
mkdir -p test_results
echo "✅ Directories created"
echo ""

# 8. Verify installation
echo "▶ Verifying installation..."
python3 -c "from secom import SeCom; print('✅ secom')" 2>/dev/null || echo "⚠️  secom import failed"
python3 -c "from llmlingua import PromptCompressor; print('✅ llmlingua')" 2>/dev/null || echo "⚠️  llmlingua import failed"
python3 -c "import chromadb; print('✅ chromadb')" 2>/dev/null || echo "⚠️  chromadb not found"
python3 -c "import langchain; print('✅ langchain')" 2>/dev/null || echo "⚠️  langchain not found"
python3 -c "import sentence_transformers; print('✅ sentence_transformers')" 2>/dev/null || echo "⚠️  sentence_transformers not found"
python3 -c "import rouge_score; print('✅ rouge_score')" 2>/dev/null || echo "⚠️  rouge_score not found"
python3 -c "import bert_score; print('✅ bert_score')" 2>/dev/null || echo "⚠️  bert_score not found"
echo ""

# 9. Summary
echo "============================================================"
echo "✅ Setup completed!"
echo "============================================================"
echo ""
echo "📋 What was done:"
echo "  ✓ SeCom package installed"
echo "  ✓ LLMLingua for compression installed"
echo "  ✓ Dependencies installed (langchain, chromadb, etc.)"
echo "  ✓ Dataset downloaded/verified"
echo "  ✓ Directories created"
echo ""
echo "📝 Next steps:"
echo ""
echo "  1. (Required) Start self-hosted LLM server:"
echo "     vllm serve Qwen/Qwen3-8B --port 8001"
echo ""
echo "  2. Run quick test:"
echo "     ./quick_test.sh (if available)"
echo ""
echo "  3. Or run full benchmark:"
echo "     ./full_benchmark_locomo.sh"
echo "     ./full_benchmark_longmemeval.sh"
echo ""
echo "💡 Tips:"
echo "  - SeCom uses self-hosted LLM for segmentation (no OpenAI key needed)"
echo "  - Edit *.sh files to change LLM model/server settings"
echo "  - Check worker_logs/ if parallel indexing fails"
echo "  - Results saved to benchmark_results/"
echo ""
