mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))

DATA_DIR ?= $(dir $(mkfile_path))data

VOL_NAME ?= /data
VOL ?= -v $(DATA_DIR):$(VOL_NAME)

PREPROCESS_FLAGS ?= -it
TRAIN_FLAGS ?= -it

# Input and output to preprocessing
PREP_INPUT_FILE ?= tiny-shakespeare.txt
PREP_OUTPUT_FILE ?= tiny-shakespeare

# Training arguments for train.lua
TRAIN_INPUT_FILE ?= $(PREP_OUTPUT_FILE)
TRAIN_ARGS ?= -model_type rnn -num_layers 3 -rnn_size 256 -checkpoint_name '$(VOL_NAME)/checkpoints/checkpoint'

MODEL_NAME ?= invalid_model.t7

.PHONY: build-torch build-preprocess build-server run-preprocess train preprocess serve

build-torch:
	docker build -t diku-hcc/torch -f Dockerfile.torch .

build-server: 
	docker build -t diku-hcc/server -f Dockerfile.server .

build-preprocess:
	docker build -t diku-hcc/preprocess -f Dockerfile.preprocess .

preprocess-bash:
	docker run --runtime=nvidia $(VOL) -it diku-hcc/preprocess:latest bash

preprocess:
	docker run $(VOL) $(PREPROCESS_FLAGS) diku-hcc/preprocess:latest python /opt/server/preprocess.py --input_txt "$(VOL_NAME)/$(PREP_INPUT_FILE)" --output_h5 "$(VOL_NAME)/generated/$(PREP_OUTPUT_FILE).h5" --output_json "$(VOL_NAME)/generated/$(PREP_OUTPUT_FILE).json"

train:
	docker run --runtime=nvidia $(VOL) $(TRAIN_FLAGS) diku-hcc/server:latest th /opt/server/train.lua -input_h5 "$(VOL_NAME)/generated/$(TRAIN_INPUT_FILE).h5" -input_json "$(VOL_NAME)/generated/$(TRAIN_INPUT_FILE).json" $(TRAIN_ARGS)

serve:
	docker run --runtime=nvidia -p 8080:8080 $(VOL) diku-hcc/server:latest th /opt/server/server.lua -checkpoint '$(VOL_NAME)/checkpoints/$(MODEL_NAME)'
