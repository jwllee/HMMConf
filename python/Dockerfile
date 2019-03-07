FROM ubuntu:18.04
LABEL maintainer "Wai Lam Jonathan Lee <jonathan.wailam.lee@gmail.com>"

ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	git \
	vim \
	make \
	python3-pip \
	python3-dev \
	build-essential

ENV HOME=/root
WORKDIR $HOME

# setup dotfiles
RUN git clone --recursive https://github.com/jwllee/dotfiles.git \
	&& cd dotfiles && make 

# setup vim
RUN git clone --recursive https://github.com/jwllee/.vim.git .vim \
	&& ln -sf /.vim/vimrc /.vimrc \
	&& cd $HOME/.vim \
	&& git submodule update --init

# standard locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
# set locales UTF-8
RUN apt-get update && apt-get install -y \
	locales && locale-gen en_US.UTF-8

ENV LANG en_US.UTF-8

# packages for PM4Py
RUN apt-get update && apt-get install -y \
	graphviz \
	libblas-dev \
	liblapack-dev \
	libsuitesparse-dev \
	python3-tk

WORKDIR $HOME/slac/code

# install requirements
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
