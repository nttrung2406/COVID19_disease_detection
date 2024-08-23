# Use the official NVIDIA PyTorch container as the base image
FROM nvidia/cuda:12.2.0-cudnn9-devel-ubuntu20.04

# Set environment variables for Python and CUDA
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV CUDA_HOME /usr/local/cuda

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /workspace
COPY . .
CMD ["python3.9", "training.py"]
