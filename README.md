This is a Dockerized and slightly modified version of [`torch-rnn-server`](https://github.com/robinsloan/torch-rnn-server) that allows you to do preprocessing in Python, run Torch interactively, train a model and serve the model all via docker containers. A Makefile is provided for easier access to building
the necessary containers and running them with sensible arguments.

Thank you to Robin Sloan for his awesome work on `torch-rnn-server` and to Justin Johnson for [`torch-rnn`](https://github.com/jcjohnson/torch-rnn) that runs below all of this wrapping - the real meat
is in `torch-rnn`!

### Installation

Because everything runs inside Docker containers, you do not need Torch, Lua, HDF5, etc.
Instead you will just need to install the following:

- Docker (e.g, [`Docker CE`](https://www.docker.com/community-edition))
- NVIDIA Docker runtime (if you want to use CUDA) [NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Note: _The version of your CUDA driver should match the CUDA runtime base image!_ - The default
used in this repository is CUDA 9.1. If you are running an earlier or later version of CUDA,
then you should modify the `FROM` statement in `Dockerfile.base` to reflect the correct version
from [`nvidia/cuda`](https://hub.docker.com/r/nvidia/cuda/).

### General concepts

In essence this fork does a few things:

- Re-organize the code into the `src/` and `docker-scripts/` folders.
- Provide 4 Dockerfiles, for running 3 different kinds of containers: preprocessing, torch/training, and a webserver.
- Provide a Makefile for easier building and running of Docker containers.

The Docker containers run using the NVIDIA Docker runtime by default, and they all mount
a data directory to `/data` inside the container for data manipulation / permanence.

### Dockerfiles

#### Dockerfile.base
`Dockerfile.base` is a base image running NVIDIA's official CUDA 9.1+CudNN7 development
image (which runs Ubuntu 16.04), with Torch built for CUDA support and a number of relevant Lua
libraries that `torch-rnn-server` uses.

This Docker image, by far, takes the longest to build -- but is also the least likely to need
changes.

#### Dockerfile.torch
`Dockerfile.torch` builds on top of `Dockerfile.base` and adds an `/opt/torch` directory with
the Lua sourcecode from `torch-rnn-server` and a script for easier training of models. This
image is both used as a base for another image, and useful for training models.

#### Dockerfile.server
`Dockerfile.server` builds on top of `Dockerfile.torch` and adds the Lua-based
webserver `server.lua` from `torch-rnn-server`, along with exposing port 8080 for
contact with the host machine.

#### Dockerfile.preprocessing
A simple Python2.7 image for data manipulation, able to run the preprocessing script from
`torch-rnn-server`.

### Data layout and expectations

Containers expect there to be a `/data` mount available, and several scripts or lua files expect
`/data/generated` to hold `.h5` and `.json` generated files, and `/data/checkpoints` to hold model
checkpoints. Everything has been modified to output to the correct folders according to
this.

The following steps assume we are trying to preprocess, train and then serve the `tiny-shakespeare` dataset,
which starts as a `.txt` file we want to train on. We expect the file to be at `./data/tiny-shakespeare.txt`

### Preprocessing

The RNN model expects input in the form of a pair of `.h5` and `.json` file(s) representing the training
data. The `preprocess.py` script from `torch-rnn-server` facilitates this.

First, build the preprocessing container: `make build-preprocess`

Now to preprocess e.g., the `tiny-shakespeare.txt` dataset you can either go into the container and run 
`preprocess.py` yourself:
```
make preprocess
python preprocess.py --input_txt=/data/tiny-shakespeare.txt --output_h5=/data/generated/tiny-shakespeare.h5 --output_json=/data/generated/tiny-shakespeare.json
```

Or run the convenience script `./scripts/tiny-shakespeare/preprocess.sh` outside Docker which does
the same thing.

This will generate `/data/generated/tiny-shakespeare.h5` and `/data/generated/tiny-shakespeare.json`.

### Training

Training the RNN model requires Torch and Lua-dependencies, so you must use the torch Dockerfile for this.

First build the container: `make build-torch`

Building the image may take a long time, if you haven't built the base image (`Dockerfile.base`) yet. This is normal, 
but you will likely only need to build the base image once.

Then either go into the container and run `train.py` yourself:
```
make torch
cd /opt/torch
python scripts/train.py --input_json=/data/generated/tiny-shakespeare.json --input_h5=/data/generated/tiny-shakespeare.h5 --checkpoint_name=/data/checkpoints/tiny_shakespeare
```

Or run the convenience script `./scripts/tiny-shakespeare/train.sh`. This will spawn a detached Docker instance that
will start training a model. To inspect it you can use e.g., `docker attach`, or if you want it to be in interactive mode
instead, pass `DOCKER_FLAGS=-i`, i.e. `DOCKER_FLAGS=-i ./scripts/tiny-shakespeare/train.sh`.

### Serving

Once done training, you can serve any `.t7` model using the Lua webserver this project comes with.

First, build the server container: `make build-server`

Then supposing the `.t7` model file you want to run is called `tiny_shakespeare_250000.t7` then you can:

Either run `MODEL_NAME=tiny_shakespeare_250000.t7 make serve`, or `./scripts/tiny-shakespeare/serve.sh tiny_shakespeare_250000.t7`
which is just a shortcut for that Makefile command.

### Specifying a different data directory

By default, the project mounts the local `torch-rnn-server/data` folder as `/data` in the Docker containers, but you will
likely want to keep your data out of this project repository and somewhere else. The `Makefile` allows you to override the
`DATA_DIR` environment variable, which should point to the full path of the directory you want mounted as `/data` in the 
containers. 

You have 3 options:

1. If you want to run `Makefile` tasks directly, then redefine `DATA_DIR` in the `Makefile` to point to the path you want it to.
2. If you want to run via the `scripts/xxx/train|preprocess|serve.sh` files, then modify the exported `DATA_DIR` in `scripts/data-dir.sh`.
3. Specify `DATA_DIR` manually in the `Makefile` tasks you run.

### Should I use this over the original `torch-rnn-server`?

This fork exists primarily to make it (much) easier to get up and running, without a _lot_ of
system-specific dependencies all over the place. Even then, it relies on a specific NVIDIA-provided
CUDA 9.1 image. If you think reproducible environments are a good thing, and you want to run CUDA-backed
model training in containers, then this fork might be of interest to you. In theory if your machine
is running the CUDA 9.1 driver, you should be able to just pull this repository down, build the images
and get to it.

If you do not intend to run the Docker containers, and you are not using CUDA, then there is no
reason at all to use this fork. If you would like to use Docker, but not CUDA, I've added a small
section about that below.

### Using Docker but not CUDA
Either rename `Makefile.no_cuda` to `Makefile`, or use `make -f Makefile.no_cuda` with all of the same make targets 
as above. The `Makefile.no_cuda` file uses all of the `Dockerfile.xxx_no_cuda` files, which are an attempt to strip
anything CUDA-related from the Docker images.

Very little testing has been done on the `_no_cuda` Dockerfiles, so YMMV. They are based off plain Ubuntu 16.04 instead
of the NVIDIA CUDA images.

### Original readme from robinsloan/torch-rnn-server
This is a small server that works with the Atom package [`rnn-writer`](https://github.com/robinsloan/rnn-writer) to provide responsive, inline "autocomplete" powered by a recurrent neural network trained on a corpus of sci-fi stories, or another corpus of your choosing.

More accurately: it's a set of shims laid beneath Justin Johnson's indispensable `torch-rnn` package.

I explain what this project is all about [here](https://www.robinsloan.com/note/writing-with-the-machine); it's probably worth reading that before continuing.

### Installation

There are a couple of different ways to get `torch-rnn-server` running, but no matter what, you'll need to install Torch, the scientific computing framework that powers the whole operation. Those instructions are below, in the original `torch-rnn` README.

After completing those steps, you'll need to install Ben Glard's [`waffle`](https://github.com/benglard/waffle) project to power the web server:

```
luarocks install https://raw.githubusercontent.com/benglard/htmlua/master/htmlua-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/benglard/waffle/master/waffle-scm-1.rockspec
```

### Training and models

After installing Torch and all of `torch-rnn`'s dependencies, you can train a model on a corpus of your choosing; those instructions are below, in the original `torch-rnn` README. **Alternatively, you can download a pre-trained model** derived from ~150MB of old sci-fi stories:

```
cd checkpoints
wget https://www.dropbox.com/s/vdw8el31nk4f7sa/scifi-model.zip
unzip scifi-model.zip
```

(You can read a bit more about the corpus and find a link to download it [here](https://www.robinsloan.com/note/writing-with-the-machine).)

### Running the server

Finally! You can start the server with:

```
th server.lua
```

Or, if that gives you an error but your system supports OpenCL (this is the case with many Macs) you can start the server with:

```
th server.lua -gpu_backend opencl
```

Or, if you're still getting an error, you can run in CPU mode:

```
th server.lua -gpu -1
```

Once the server is running, try

```
curl "http://0.0.0.0:8080/generate?start_text=It%20was%20a%20dark&n=3"
```

If you see a JSON response offering strange sentences, it means everything is working, and it's onward to [`rnn-writer`](https://github.com/robinsloan/rnn-writer)!

Standard `torch-rnn` README continues below.

# torch-rnn
torch-rnn provides high-performance, reusable RNN and LSTM modules for torch7, and uses these modules for character-level
language modeling similar to [char-rnn](https://github.com/karpathy/char-rnn).

You can find documentation for the RNN and LSTM modules [here](doc/modules.md); they have no dependencies other than `torch`
and `nn`, so they should be easy to integrate into existing projects.

Compared to char-rnn, torch-rnn is up to **1.9x faster** and uses up to **7x less memory**. For more details see
the [Benchmark](#benchmarks) section below.


# Installation

## System setup

**`torch-rnn-server note`: You can skip this if you're using a pretrained model.**

You'll need to install the header files for Python 2.7 and the HDF5 library. On Ubuntu you should be able to install
like this:

```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
```

## Python setup

**`torch-rnn-server note`: You can skip this if you're using a pretrained model.**

The preprocessing script is written in Python 2.7; its dependencies are in the file `requirements.txt`.
You can install these dependencies in a virtual environment like this:

```bash
virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment
```

## Lua setup

**`torch-rnn-server note`: You can't skip this :(**

The main modeling code is written in Lua using [torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html#_). You'll need the following Lua packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/optim](https://github.com/torch/optim)
- [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)

After installing torch, you can install / update these packages by running the following:

```bash
# Install most things using luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson

# We need to install torch-hdf5 from GitHub

**`torch-rnn-server note`: You can skip this if you're using a pretrained model.**

git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec
```

### CUDA support (Optional)

**`torch-rnn-server note`: If you skip this, everything will be slowww :(**

To enable GPU acceleration with CUDA, you'll need to install CUDA 6.5 or higher and the following Lua packages:
- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

You can install / update them by running:

```bash
luarocks install cutorch
luarocks install cunn
```

## OpenCL support (Optional)
To enable GPU acceleration with OpenCL, you'll need to install the following Lua packages:
- [cltorch](https://github.com/hughperkins/cltorch)
- [clnn](https://github.com/hughperkins/clnn)

You can install / update them by running:

```bash
luarocks install cltorch
luarocks install clnn
```

## OSX Installation
Jeff Thompson has written a very detailed installation guide for OSX that you [can find here](http://www.jeffreythompson.org/blog/2016/03/25/torch-rnn-mac-install/).

**`torch-rnn-server note`: You can STOP HERE if you're using a pretrained model.**

# Usage
To train a model and use it to generate new text, you'll need to follow three simple steps:

## Step 1: Preprocess the data
You can use any text file for training models. Before training, you'll need to preprocess the data using the script
`scripts/preprocess.py`; this will generate an HDF5 file and JSON file containing a preprocessed version of the data.

If you have training data stored in `my_data.txt`, you can run the script like this:

```bash
python scripts/preprocess.py \
  --input_txt my_data.txt \
  --output_h5 my_data.h5 \
  --output_json my_data.json
```

This will produce files `my_data.h5` and `my_data.json` that will be passed to the training script.

There are a few more flags you can use to configure preprocessing; [read about them here](doc/flags.md#preprocessing)

## Step 2: Train the model
After preprocessing the data, you'll need to train the model using the `train.lua` script. This will be the slowest step.
You can run the training script like this:

```bash
th train.lua -input_h5 my_data.h5 -input_json my_data.json
```

This will read the data stored in `my_data.h5` and `my_data.json`, run for a while, and save checkpoints to files with
names like `cv/checkpoint_1000.t7`.

You can change the RNN model type, hidden state size, and number of RNN layers like this:

```bash
th train.lua -input_h5 my_data.h5 -input_json my_data.json -model_type rnn -num_layers 3 -rnn_size 256
```

By default this will run in GPU mode using CUDA; to run in CPU-only mode, add the flag `-gpu -1`.

To run with OpenCL, add the flag `-gpu_backend opencl`.

There are many more flags you can use to configure training; [read about them here](doc/flags.md#training).

## Step 3: Sample from the model
After training a model, you can generate new text by sampling from it using the script `sample.lua`. Run it like this:

```bash
th sample.lua -checkpoint cv/checkpoint_10000.t7 -length 2000
```

This will load the trained checkpoint `cv/checkpoint_10000.t7` from the previous step, sample 2000 characters from it,
and print the results to the console.

By default the sampling script will run in GPU mode using CUDA; to run in CPU-only mode add the flag `-gpu -1` and
to run in OpenCL mode add the flag `-gpu_backend opencl`.

There are more flags you can use to configure sampling; [read about them here](doc/flags.md#sampling).

# Benchmarks
To benchmark `torch-rnn` against `char-rnn`, we use each to train LSTM language models for the tiny-shakespeare dataset
with 1, 2 or 3 layers and with an RNN size of 64, 128, 256, or 512. For each we use a minibatch size of 50, a sequence
length of 50, and no dropout. For each model size and for both implementations, we record the forward/backward times and
GPU memory usage over the first 100 training iterations, and use these measurements to compute the mean time and memory
usage.

All benchmarks were run on a machine with an Intel i7-4790k CPU, 32 GB main memory, and a Titan X GPU.

Below we show the forward/backward times for both implementations, as well as the mean speedup of `torch-rnn` over
`char-rnn`. We see that `torch-rnn` is faster than `char-rnn` at all model sizes, with smaller models giving a larger
speedup; for a single-layer LSTM with 128 hidden units, we achieve a **1.9x speedup**; for larger models we achieve about
a 1.4x speedup.

<img src='imgs/lstm_time_benchmark.png' width="800px">

Below we show the GPU memory usage for both implementations, as well as the mean memory saving of `torch-rnn` over
`char-rnn`. Again `torch-rnn` outperforms `char-rnn` at all model sizes, but here the savings become more significant for
larger models: for models with 512 hidden units, we use **7x less memory** than `char-rnn`.

<img src='imgs/lstm_memory_benchmark.png' width="800px">


# TODOs
- Get rid of Python / JSON / HDF5 dependencies?
