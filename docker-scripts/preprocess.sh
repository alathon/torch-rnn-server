#!/bin/bash

# Pass input file as $1, output dir as $2, output filename as $3 (without extension)
cd /opt/server
mkdir -p "$2"
virtualenv -p python2.7 .env
source .env/bin/activate
pip install -r requirements.txt
python /opt/server/scripts/preprocess.py \
  --input_txt "$1" \
  --output_h5 "$2/$3.h5" \
  --output_json "$2/$3.json"
