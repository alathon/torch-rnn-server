#!/bin/sh

CMD="python preprocess.py \
	--input_txt=/data/tiny-shakespeare.txt \
	--output_h5=/data/generated/tiny-shakespeare.h5 \
	--output_json=/data/generated/tiny-shakespeare.json" make preprocess

