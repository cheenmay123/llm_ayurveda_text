#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
conda develop "${SCRIPT_DIR}"