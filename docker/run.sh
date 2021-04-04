#!/bin/sh
#
# Run the docker container.

. docker/env.sh
docker stop $CONTAINER_NAME
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p $HOST_PORT:$CONTAINER_PORT \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME
docker exec \
  -dit \
  $CONTAINER_NAME sh /workspace/docker/init.sh