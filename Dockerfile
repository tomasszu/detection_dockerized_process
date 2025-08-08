FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    git wget unzip \
    && apt-get clean

# Install Python packages
RUN pip3 install --upgrade pip
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Set working directory
WORKDIR /app
COPY . /app


CMD ["python3", "main.py"]