#!/bin/bash

# Source the exported DATA_DIR from ../data-dir.sh
source $(dirname $(readlink -f "$0"))/../data-dir.sh

MODEL_NAME=$1 make serve
