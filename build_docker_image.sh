#!/bin/sh
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -f docker/Dockerfile \
  -t dnn_template \
  .
