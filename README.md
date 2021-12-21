# dlg-nifty-components

[![codecov](https://codecov.io/gh/ICRAR/dlg-nifty-components/branch/main/graph/badge.svg?token=dlg-nifty-components_token_here)](https://codecov.io/gh/ICRAR/dlg-nifty-components)
[![CI](https://github.com/ICRAR/dlg-nifty-components/actions/workflows/main.yml/badge.svg)](https://github.com/ICRAR/dlg-nifty-components/actions/workflows/main.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

dlg-nifty-components contains a collection of cpu and gpu nifty gridding/degridding implementations for radio interferometry datasets.

## Installation

There are multiple options for the installation, depending on how you are intending to run the DALiuGE engine, directly in a virtual environment (host) or inside a docker container. You can also install it either from PyPI (latest released version).

## Install it from PyPI

### Prerequisites

The following packages are required before installation such that dlg-nifty-components can compile it's dependencies:
* g++
* python3-dev
* numpy
* cuda-libraries-dev

### DALiuGE Engine Python App in Docker Container

For development purposes it is preferable to install and run dlg-nifty-components as a python app for fast recompilation of dynamically linked binaries. To do this you must first install a cuda version in daliuge-common or daliuge-engine followed by python-dev, numpy and g++. daliuge-nifty-components will then compile and install wagg for cuda acceleration.

daliuge-common/Dockerfile.dev

```bash
RUN apt install -y wget gnupg2 software-properties-common
RUN mkdir -p /code && cd /code && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt update

RUN DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
    cuda-minimal-build-11-2 cuda-libraries-11-2 cuda-libraries-dev-11-2 && \
    ln -s /usr/local/cuda-11.2 /usr/local/cuda && \
    ln -s /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib && \
    ln -s /usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda/include
```

daliuge-engine/Dockerfile.dev

```bash
RUN apt install -y python3-dev g++
```

run_engine.sh

```bash
# append configured nvidia-docker arguments here, e.g.
DOCKER_OPTS=$DOCKER_OPTS --gpus=all --privileged
```

dlg-nifty-components may be installed before or after DALiuGE engine is running:

```bash
docker exec -t daliuge-engine bash -c 'pip install --prefix=$DLG_ROOT/code dlg_nifty_components'
```

### EAGLE Palette

An EAGLE .palette file can be conveniently generated locally using command:

```bash 
bash ./build_palatte.sh
```

## Usage

### Python

For example the MS2DirtyApp component will be available to the engine when you specify

```python
from daliuge_component_nifty import MS2DirtyApp

MS2DirtyApp('a','a')
```

in the AppClass field of a Python Branch component. The EAGLE palette associated with these components are also generated and can be loaded directly into EAGLE. In that case all the fields are correctly populated for the respective components.

### DALiuGE Docker App

Optionallyb uild the container image for use as a daliuge docker app:

```bash
docker build -t dlg-nifty-components -f ./Containerfile .
```
