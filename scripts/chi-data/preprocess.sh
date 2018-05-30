#!/bin/bash

# Source the exported DATA_DIR from ../data-dir.sh
source $(dirname $(readlink -f "$0"))/../data-dir.sh

CMD="python preprocess.py \
	--input_txt=/data/chi-intros.txt \
	--output_h5=/data/generated/chi-intros.h5 \
	--output_json=/data/generated/chi-intros.json" make preprocess

