FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
MAINTAINER Martin Gruenbaum "martin@itsolveonline.net"

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# Required packages
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    git \
    libhdf5-dev \
    sudo \
    software-properties-common \
    wget

# Install Torch
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
    bash install-deps 

RUN cd /root/torch && \
    TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__" ./install.sh

ENV PATH=/root/torch/install/bin:$PATH

# Lua dependencies
RUN luarocks install cutorch && \
    luarocks install cunn && \
    luarocks install cudnn && \
    luarocks install nn && \
    luarocks install optim && \
    luarocks install lua-cjson && \
    luarocks install https://raw.githubusercontent.com/benglard/htmlua/master/htmlua-scm-1.rockspec && \
    luarocks install https://raw.githubusercontent.com/benglard/waffle/master/waffle-scm-1.rockspec

# Install HDF5
RUN git clone https://github.com/deepmind/torch-hdf5 && \
  cd torch-hdf5 && \
  luarocks make hdf5-0-0.rockspec 

RUN mkdir /opt/server
ADD . /opt/server
RUN chmod +x -R /opt/server/docker-scripts

EXPOSE 8080
WORKDIR /opt/server
CMD ["/bin/bash"]
# Pre-trained model
# RUN wget https://www.dropbox.com/s/vdw8el31nk4f7sa/scifi-model.zip
