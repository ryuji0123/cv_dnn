#!/bin/sh
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
ARG_DIR="$(pwd)/.arg"
LOG_DIR="$(pwd)/.log"
ARGS_FILE="${ARG_DIR}/${TIMESTAMP}.yaml"
TRAIN_LOG_FILE="${LOG_DIR}/${TIMESTAMP}.txt"
mkdir -p $ARG_DIR
mkdir -p $LOG_DIR

export CUDA_VISIBLE_DEVICES=3
nohup python -u train.py \
  --args_file_path $ARGS_FILE \
  --train_log_file_path $TRAIN_LOG_FILE \
  >> $TRAIN_LOG_FILE &
tail -f $TRAIN_LOG_FILE
