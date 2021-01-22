#!/bin/sh
docker run \
  -dit \
  --gpus all \
  --name dnn_template \
  --rm \
  --shm-size=2gb \
  -v ~/dnn_template:/workspace \
  dnn_template bash
