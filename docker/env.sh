#!/bin/sh
#
# Set environment variables.

export IMAGE_NAME=${USER}_cv_dnn
export CONTAINER_NAME=${USER}_cv_dnn
export MLFLOW_HOST_PORT=5000
export MLFLOW_CONTAINER_PORT=5000

if [ -e docker/env_dev.sh ]; then
  . docker/env_dev.sh
fi