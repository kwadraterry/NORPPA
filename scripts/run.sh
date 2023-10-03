#!/bin/bash

NORPPA_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
export PYTHONPATH="${PYTHONPATH}:${NORPPA_DIR}"


cd $NORPPA_DIR

script=${1/%".py"}

shift;

python "./scripts/$script.py" "$@" | tee "./output/$script.txt"