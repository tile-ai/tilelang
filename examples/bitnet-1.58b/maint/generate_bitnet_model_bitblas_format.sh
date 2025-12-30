#!/usr/bin/env bash

# retrieve the native model input and saved model directory
MODEL_DIR=$1
SAVED_MODEL_DIR=$2

# check if the model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory does not exist!"
    exit 1
fi

# if the saved model directory does not exist, create it
# if SAVED_MODEL_DIR is not provided, we do not pass it to the script
if [ -z "$SAVED_MODEL_DIR" ]; then
  python ./maint/create_bitblas_ckpt.py --model_name_or_path $MODEL_DIR
else
  if [ ! -d "$SAVED_MODEL_DIR" ]; then
    mkdir -p $SAVED_MODEL_DIR
  fi
  python ./maint/create_bitblas_ckpt.py --model_name_or_path $MODEL_DIR --saved_model_path $SAVED_MODEL_DIR
fi

# get the realpath of the saved model directory
SAVED_MODEL_DIR=$(realpath $SAVED_MODEL_DIR)

# cp files
cp $MODEL_DIR/quantize_config.json $SAVED_MODEL_DIR/
cp $MODEL_DIR/tokenizer.json $SAVED_MODEL_DIR/
cp $MODEL_DIR/tokenizer.model $SAVED_MODEL_DIR/
cp $MODEL_DIR/tokenizer_config.json $SAVED_MODEL_DIR/

echo "Model has been converted and save to $SAVED_MODEL_DIR"
