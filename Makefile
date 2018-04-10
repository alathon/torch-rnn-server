mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))

DATA_DIR ?= $(dir $(mkfile_path))data

IMAGE_NAME ?= torch-rnn-server

VOL_NAME ?= /data
VOL ?= -v $(DATA_DIR):$(VOL_NAME)

PREPROCESS_FLAGS ?= -it
TRAIN_FLAGS ?= -it

# Input and output to preprocessing
PREP_INPUT_FILE ?= tiny-shakespeare.txt
PREP_OUTPUT_FILE ?= tiny-shakespeare

# Training arguments for train.lua
TRAIN_ARGS ?= -model_type rnn -num_layers 3 -rnn_size 256 -checkpoint_name '$(VOL_NAME)/checkpoints/checkpoint'

MODEL_NAME ?= invalid_model.t7

.PHONY: build run-bash train preprocess

build: Dockerfile
	docker build -t $(IMAGE_NAME) -f Dockerfile .

run-bash:
	docker run --runtime=nvidia $(VOL) -it torch-rnn-server bash

preprocess:
	docker run $(VOL) $(PREPROCESS_FLAGS) $(IMAGE_NAME) bash /opt/server/docker-scripts/preprocess.sh $(VOL_NAME)/$(PREP_INPUT_FILE) $(VOL_NAME)/generated/$(PREP_OUTPUT_FILE)

train:
	docker run --runtime=nvidia $(VOL) $(TRAIN_FLAGS) $(IMAGE_NAME) bash /opt/server/docker-scripts/train.sh $(VOL_NAME)/generated/$(PREP_OUTPUT_FILE) $(TRAIN_ARGS)

serve:
	docker run --runtime=nvidia -p 8080:8080 $(VOL) $(IMAGE_NAME) th server.lua -checkpoint '$(VOL_NAME)/checkpoints/$(MODEL_NAME)'

clean:
	docker run $(VOL) -it $(IMAGE_NAME) rm -rf $(VOL)/generated
