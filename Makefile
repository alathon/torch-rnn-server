mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))

DATA_DIR ?= $(dir $(mkfile_path))data
DOCKER_FLAGS ?=-it
MODEL_NAME ?= "invalid-model.t7"

# You probably shouldn't touch these
SERVE_CMD=th /opt/server/server.lua -checkpoint '$(MOUNT_NAME)/checkpoints/$(MODEL_NAME)'
MOUNT_NAME=/data
MOUNT=-v $(DATA_DIR):$(MOUNT_NAME)

.PHONY: build-base build-server build-torch build-preprocess preprocess server torch

build-base:
	docker build -t diku-hcc/base -f Dockerfile.base .

build-torch:
	docker build -t diku-hcc/torch -f Dockerfile.torch .

build-server:
	docker build -t diku-hcc/server -f Dockerfile.server .

build-preprocess:
	docker build -t diku-hcc/preprocess -f Dockerfile.preprocess .

torch:
	docker run --runtime=nvidia $(MOUNT) $(DOCKER_FLAGS) diku-hcc/torch:latest $(CMD)

server:
	docker run --runtime=nvidia $(MOUNT) $(DOCKER_FLAGS) diku-hcc/server:latest $(CMD)

serve:
	docker run --runtime=nvidia -p 8080:8080 $(MOUNT) $(DOCKER_FLAGS) diku-hcc/server:latest $(SERVE_CMD)

preprocess:
	docker run --runtime=nvidia $(MOUNT) $(DOCKER_FLAGS) diku-hcc/preprocess:latest $(CMD)
