#!/bin/sh
TMP_RESULTS_DIR="$(pwd)/.tmp_results"
mkdir -p $TMP_RESULTS_DIR
TRAIN_LOG_FILE="${TMP_RESULTS_DIR}/all_optuna_train_log.txt"
if [ -e $TRAIN_LOG_FILE ]; then
  rm $TRAIN_LOG_FILE
fi 

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2,3
nohup python -u train_with_optuna.py \
  --tmp_results_dir $TMP_RESULTS_DIR \
  >> $TRAIN_LOG_FILE &
sleep 1s
tail -f $TRAIN_LOG_FILE
