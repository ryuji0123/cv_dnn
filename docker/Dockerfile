FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# apt-get
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && \
		apt-get install -y \
			curl libopencv-dev lsof git sudo tmux tree vim wget && \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		rm -rf /var/cache/apk/*

# python
ENV WORK_PATH /workspace
WORKDIR $WORK_PATH
ENV PYTHONPATH "/workspace:${PYTHONPATH}"
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
COPY docker/requirements.txt $WORK_PATH/docker/
RUN pip install ${PIP_OPTIONS} -r $WORK_PATH/docker/requirements.txt && \
		pip install ${PIP_OPTIONS} -U setuptools

# duser setting
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID duser && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser && \
    adduser duser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER duser
