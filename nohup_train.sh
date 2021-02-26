#!/bin/sh
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
RESULTS_DIR="$(pwd)/results/${TIMESTAMP}"
ARGS_FILE="${RESULTS_DIR}/args.yaml"
TRAIN_LOG_FILE="${RESULTS_DIR}/log.txt"
mkdir -p $RESULTS_DIR

export CUDA_VISIBLE_DEVICES=3
nohup python -u train.py \
  --args_file_path $ARGS_FILE \
  --train_log_file_path $TRAIN_LOG_FILE \
  >> $TRAIN_LOG_FILE &
sleep 1s
tail -f $TRAIN_LOG_FILE
