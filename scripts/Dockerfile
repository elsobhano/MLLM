FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update

RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    rsync \
    wget \
    ffmpeg \
    htop \
    nano \
    libatlas-base-dev \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenblas-dev \
    libopenblas-base \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi7

RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

ENV WRKSPCE="/workspace"

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPCE/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="$WRKSPCE/miniconda3/bin:${PATH}"


RUN conda install -n base -c defaults conda python=3.10

COPY . .

RUN conda env update -n base --file environment.yml 

RUN conda clean -y --all

RUN mkdir /.cache /.config && \
    chmod 777 /.cache /.config && \
    chmod -R 777 /.cache /.config

RUN chmod -R 777 /workspace