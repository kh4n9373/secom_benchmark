# SeCom Benchmark Pipeline

Comprehensive benchmark pipeline for SeCom (Segment and Compress) memory system, implementing topical segmentation and compression-based retrieval for conversational AI.

## 🎯 Overview

This benchmark pipeline evaluates SeCom's performance on long-term conversational memory tasks using:
- **Topical Segmentation**: LLM-based conversation segmentation
- **Compression**: LLMLingua-2 based memory compression
- **Vector Retrieval**: Semantic search with compressed memory units
- **Evaluation**: Precision, Recall, F1, nDCG metrics

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/kh4n9373/secom_benchmark.git
cd secom_benchmark

# Run setup (installs dependencies, downloads datasets)
bash setup.sh
```

**Setup includes:**
- ✅ SeCom package installation
- ✅ LLMLingua compression library
- ✅ Vector stores (Chroma, FAISS)
- ✅ LoCoMo & LongMemEval datasets

### 2. Start LLM Server

SeCom requires a self-hosted LLM for segmentation:

```bash
# Using vLLM (recommended)
vllm serve Qwen/Qwen3-8B --port 8001

# Or any OpenAI-compatible API
```

### 3. Run Quick Test

```bash
# Test with 1 conversation (~7 minutes)
bash quick_test.sh
```

### 4. Run Full Benchmark

```bash
# LoCoMo dataset
bash full_benchmark_locomo.sh

# LongMemEval dataset
bash full_benchmark_longmemeval.sh
```

---

## 📊 Benchmark Results

After running, results are saved to timestamped directories:

```
{dataset}_results_YYYYMMDD_HHMMSS/
├── retrieval_results_*.json      # Retrieved memory chunks
├── retrieval_eval_*.json          # Retrieval metrics (P, R, F1, nDCG)
├── generation_eval_*.json         # Generation metrics (BLEU, ROUGE, BERTScore)
└── pipeline_metadata_*.json       # Configuration and timings
```

**Example Metrics (1 conversation, 199 questions):**

| Metric | Value | Description |
|--------|-------|-------------|
| **Retrieval @ k=10** | P=0.12, R=0.83, F1=0.20 | Memory retrieval quality |
| **Generation** | F1=0.08, BERTScore=0.86 | Answer generation quality |
| **Pipeline Time** | ~7 minutes | End-to-end execution |

---

## 🔧 Configuration Options

### Core Parameters

```bash
python3 secom_full_pipeline.py \
    <dataset_file> <memory_dir> <output_dir> \
    --max_workers 2              # Parallel indexing workers
    --llm_model Qwen/Qwen3-8B    # LLM for segmentation
    --base_url http://localhost:8001/v1  # LLM API endpoint
    --embedding_model facebook/contriever  # Embedding model
    --granularity segment        # segment/session/turn
    --compress_rate 0.75         # Compression rate (0.0-1.0)
    --top_k 100                  # Retrieval top-k
    --context_k 5                # Generation context size
    --eval_ks "3,5,10"          # Evaluation @ k values
```

### Granularity Modes

| Mode | Description | Speed | Use Case |
|------|-------------|-------|----------|
| `segment` | LLM topical segmentation | Slow (~7 min/conv) | Best quality, research |
| `session` | Session-level chunks | Fast (~16 sec/conv) | Production, fast iteration |
| `turn` | Turn-level chunks | Fastest | Baseline comparison |

**Recommendation**: Use `segment` for final results, `session` for fast development.

---

## 🚀 Performance Optimization

### Parallel Indexing

Use multiple workers to speed up indexing:

```bash
# 4 parallel workers
python3 secom_full_pipeline.py \
    data.json memory_dir output_dir \
    --max_workers 4
```

**Speedup:**
- 1 worker: ~22 hours for 200 conversations
- 4 workers: ~5.6 hours for 200 conversations
- **4x faster** (linear scaling)

### Fast Testing

```bash
# Quick test with session granularity (much faster)
python3 secom_full_pipeline.py \
    data.json memory_dir output_dir \
    --granularity session \
    --compress_rate 0.75
```

**Speed comparison (1 conversation):**
- `segment`: ~7 minutes (LLM segmentation)
- `session`: ~16 seconds (no LLM calls)
- **~26x faster**

---

## 📁 Project Structure

```
SeCom/
├── setup.sh                          # Environment setup
├── secom_process_index.py            # Memory indexing
├── secom_process_retrieve.py         # Memory retrieval
├── secom_full_pipeline.py            # Complete pipeline orchestrator
├── full_benchmark_locomo.sh          # LoCoMo launcher
├── full_benchmark_longmemeval.sh     # LongMemEval launcher
├── quick_test.sh                     # Quick test script
├── secom/                            # SeCom core library
│   ├── secom.py                      # Main SeCom class
│   ├── utils.py                      # LLM utilities
│   └── configs/
│       ├── contriever.yaml           # Contriever config
│       ├── mpnet.yaml                # MPNet config
│       └── bm25.yaml                 # BM25 config
└── worker_logs/                      # Parallel worker logs
```

---

## 🔍 Understanding Results

### Retrieval Metrics

- **Precision @ k**: How many retrieved chunks are relevant
- **Recall @ k**: How many relevant chunks were retrieved
- **F1 @ k**: Harmonic mean of precision and recall
- **nDCG @ k**: Normalized discounted cumulative gain (ranking quality)

### Generation Metrics

- **F1**: Token overlap with ground truth
- **BLEU**: N-gram precision
- **ROUGE**: Recall-oriented text similarity
- **BERTScore**: Semantic similarity using BERT embeddings

### Category Breakdown

Results are broken down by question category (1-5):
1. Factual questions about events
2. Temporal reasoning
3. Complex multi-hop
4. Opinion/preference questions
5. Contextual understanding

---

## 🐛 Troubleshooting

### Issue: Import Errors

```bash
# Reinstall dependencies
pip install langchain-community llmlingua omegaconf
```

### Issue: Slow Segmentation

```bash
# Use session granularity instead
--granularity session
```

### Issue: Out of Memory

```bash
# Reduce parallel workers
--max_workers 1

# Or use smaller embedding model
--embedding_model sentence-transformers/all-MiniLM-L6-v2
```

### Issue: LLM Connection Failed

```bash
# Check LLM server is running
curl http://localhost:8001/v1/models

# Verify base_url parameter
--base_url http://localhost:8001/v1
```

### Parallel Worker Failures

Check worker logs for detailed errors:
```bash
cat worker_logs/worker_0.log
cat worker_logs/worker_1.log
```

---

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{secom_benchmark,
  title={SeCom Benchmark Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/kh4n9373/secom_benchmark}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📝 License

This project follows the original SeCom license. See [LICENSE](LICENSE) for details.

---

## 🔗 Related Projects

- **SeCom**: [Original Repository](https://github.com/Original/SeCom)
- **LLMLingua**: [Compression Library](https://github.com/microsoft/LLMLingua)
- **LoCoMo Dataset**: [HuggingFace](https://huggingface.co/datasets/KhangPTT373/locomo)

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review worker logs in `worker_logs/`

---

**Happy Benchmarking! 🚀**
