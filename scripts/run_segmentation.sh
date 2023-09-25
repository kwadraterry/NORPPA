#!/bin/bash

NORPPA_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
export PYTHONPATH="${PYTHONPATH}:${NORPPA_DIR}"


cd $NORPPA_DIR
# export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:4096'
python ./scripts/run_segmentation.py "$@" | tee output/segmentation_out.txt