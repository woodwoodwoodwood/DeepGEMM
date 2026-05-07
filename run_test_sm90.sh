#!/bin/bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

echo "=========================================="
echo "  DeepGEMM SM90 Compatible Tests"
echo "=========================================="
echo ""

# 1. FP8 GEMM tests (single GPU)
echo "[1/4] Running FP8/FP4 GEMM tests..."
python tests/test_fp8_fp4.py

# 2. BF16 GEMM tests (single GPU)
echo "[2/4] Running BF16 GEMM tests..."
python tests/test_bf16.py

# 3. SF layout tests (single GPU)
echo "[3/4] Running SF layout tests..."
python tests/test_layout.py

# 4. Attention tests (single GPU)
echo "[4/4] Running attention tests..."
python tests/test_attention.py

echo ""
echo "All SM90 tests completed!"
