#!/bin/sh

# Set docker to run in detached mode, and run training script
DOCKER_FLAGS="-d" CMD="python /opt/torch/docker-scripts/train.py \
	--input_json=/data/generated/chi-intros.json \
	--input_h5=/data/generated/chi-intros.h5 \
	--checkpoint_name=/data/checkpoints/checkpoint \
	--overrides -rnn_size 256 -model_type lstm -num_layers 3 -checkpoint_every 10000" \
	make torch

