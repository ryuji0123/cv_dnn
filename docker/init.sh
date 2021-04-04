#!/bin/sh
#
# Process to run after the docker container is started.

. docker/env.sh
mlflow ui --host 0.0.0.0 -p $MLFLOW_CONTAINER_PORT