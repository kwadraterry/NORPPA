#!/bin/bash

NORPPA_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
export PYTHONPATH="${PYTHONPATH}:${NORPPA_DIR}"


cd $NORPPA_DIR

python ./scripts/test_extractors.py "$@" | tee output/test_extractors_whaleshark_pie_split.txt