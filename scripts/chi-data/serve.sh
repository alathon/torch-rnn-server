#!/bin/sh

# Note: model name is just the name of the t7 file, not the full path,
# e.g. checkpoint_10000.t7 not /data/checkpoints/checkpoint_10000.t7
# So you could run this via ./scripts/serve.sh checkpoint_10000.t7
# for /data/checkpoints/checkpoint_10000.t7 to be served as the model
MODEL_NAME=$1 make serve

