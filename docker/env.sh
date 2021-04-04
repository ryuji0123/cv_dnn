#!/bin/sh
#
# Set environment variables.

export IMAGE_NAME=${USER}_cv_dnn
export CONTAINER_NAME=${USER}_cv_dnn
export HOST_PORT=5000
export CONTAINER_PORT=5000

if [ -e docker/env_dev.sh ]; then
  . docker/env_dev.sh
fi