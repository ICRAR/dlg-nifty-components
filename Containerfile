FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
# FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04
# FROM ubuntu:20.04
# FROM python:3.7-alpine
RUN apt update && apt install -y git g++ python3-pip python3-numpy
# RUN apk add --no-cache git g++ python3-numpy
COPY . /app
WORKDIR /app
RUN pip install .
CMD ["dlg_nifty_components"]
