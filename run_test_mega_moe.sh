#!/bin/bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

echo "Running MegaMoE test on GPU 3,4..."
python tests/test_mega_moe.py --num-processes 2
