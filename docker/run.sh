#!/bin/sh
. docker/env.sh
docker stop $CONTAINER_NAME
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p 6008:6008 \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME
