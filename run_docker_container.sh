#!/bin/sh
CONTAINER_NAME='cv_dnn'
docker stop $CONTAINER_NAME
docker run \
  -dit \
  --gpus all \
  --name $CONTAINER_NAME\
  -p 6008:6008 \
  --rm \
  --shm-size=2gb \
  -v ~/cv_dnn:/workspace \
  cv_dnn bash
