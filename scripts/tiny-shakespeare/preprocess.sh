#!/bin/bash

# Source the exported DATA_DIR from ../data-dir.sh
source $(dirname $(readlink -f "$0"))/../data-dir.sh

CMD="python preprocess.py \
	--input_txt=/data/tiny-shakespeare.txt \
	--output_h5=/data/generated/tiny-shakespeare.h5 \
	--output_json=/data/generated/tiny-shakespeare.json" make preprocess

