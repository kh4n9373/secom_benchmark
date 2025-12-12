#!/bin/bash
# Quick test - 20 conversations only
# Run from: amem/ directory

cd "$(dirname "$0")"

echo "⚡ Running quick test (20 conversations) with A-mem..."
echo ""

# Use timestamped directories to avoid permission issues
timestamp=$(date +%Y%m%d_%H%M%S)

# First, create a small test dataset (20 conversations)
python3 -c "
import json
with open('data/locomo/processed_data/locomo_processed_data.json', 'r') as f:
    data = json.load(f)
with open('data/locomo/processed_data/locomo_small_test.json', 'w') as f:
    json.dump(data[:1], f, indent=2)
print('Created test dataset with 20 conversations')
"

python3 secom_full_pipeline.py \
    data/locomo/processed_data/locomo_small_test.json \
    test_memory_${timestamp} \
    test_results_${timestamp} \
    --max_workers 1 \
    --llm_model Qwen/Qwen3-8B \
    --api_key dummy \
    --base_url http://localhost:8001/v1 \
    --embedding_model facebook/contriever \
    --granularity segment \
    --compress_rate 0.75 \
    --top_k 100 \
    --context_k 5 \
    --eval_ks "3,5,10" \
    --disable_thinking

echo ""
echo "✅ Quick test completed!"
echo "📂 Results saved to: test_results_${timestamp}/"

