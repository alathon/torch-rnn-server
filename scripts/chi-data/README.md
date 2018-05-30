The preprocessing script expects there to be a
chi-intros.txt file at /torch-rnn-server/data/chi-intros.txt

### Preprocessing

To preprocess the data, run ./scripts/chi-data/preprocess.sh
This produces an .h5 and .json file from the .txt file.

### Training

To train the model, run ./scripts/chi-data/train.sh
This will run a detached-mode Docker container that will
train a 256-node 3-layer LSTM, with checkpoints every 10000
iterations.

### Serving

To serve a model, run `./scripts/chi-data/serve.sh checkpoint_XXXXX.t7`
where XXXX = the iteration checkpoint you want to run

Note that the server must be killed in a separate window by docker kill
