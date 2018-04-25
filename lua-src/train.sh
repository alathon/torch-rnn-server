#!/bin/bash

# Pass filename without extension as $1, and
# training arguments (eg -model_type, -num_layers, -rnn_size, -gpu, etc) as $2

cd /opt/server
th train.lua -input_h5 $1.h5 -input_json $1.json ${@:2}
