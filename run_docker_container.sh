#!/bin/sh
docker run \
  -dit \
  --gpus all \
  --name dnn\
  --rm \
  --shm-size=2gb \
  -v ~/dnn:/workspace \
  dnn zsh
