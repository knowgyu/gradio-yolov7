FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN rm -f /etc/apt/source.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y --no-install-recommends\
    vim \
    wget \
    curl \
    apt-utils \
    tree \
    ca-certificates \
    sudo \
    build-essential \
    python3-pip \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \         
    libxext6 \       
    libxrender1 \    
    python-is-python3 \
    language-pack-ko && \
    rm -rf /var/lib/apt/lists/* && \
    conda install ffmpeg

# Set up time zone
ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Copy requirements.txt and install Python modules
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /gradio

CMD [ "/bin/bash" ]
