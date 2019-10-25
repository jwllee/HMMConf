FROM ubuntu:18.04
LABEL maintainer "Wai Lam Jonathan Lee <jonathan.wailam.lee@gmail.com>"

ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV UNAME hmmconf
ENV HOME=/home/${UNAME}
WORKDIR $HOME

RUN apt-get update && apt-get install -y \
	wget \
	git \
	vim \
	make \
	python3-pip \
	python3-dev \
	build-essential \
	eog \
	libcanberra-gtk-module \
	libcanberra-gtk3-module 

# setup dotfiles
RUN git clone --recursive https://github.com/jwllee/dotfiles.git \
	&& cd dotfiles && make 

# setup vim
RUN git clone --recursive https://github.com/jwllee/.vim.git .vim \
	&& ln -sf /.vim/vimrc /.vimrc \
	&& cd $HOME/.vim \
	&& git submodule update --init

RUN export UNAME=$UNAME UID=1000 GID=1000 && \
	mkdir -p "/home/${UNAME}" && \
	echo "${UNAME}:x:${UID}:${GID}:${UNAME} User,,,:/home/${UNAME}:/bin/bash" >> /etc/passwd && \
	echo "${UNAME}:x:${UID}:" >> /etc/group && \
	mkdir -p /etc/sudoers.d && \
	echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${UNAME} && \
	chmod 0440 /etc/sudoers.d/${UNAME} && \
	chown ${UID}:${GID} -R /home/${UNAME} && \
	gpasswd -a ${UNAME} audio

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

WORKDIR $HOME/code

RUN echo 'alias python=python3' >> $HOME/.bashrc \
	&& echo 'alias pip=pip3' >> $HOME/.bashrc 

# install requirements
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

RUN apt-get update && apt-get install -y \
	unzip 

USER $UNAME

# check that hmmlearn was installed
RUN python3 -c "import hmmlearn; print('hmmlearn version: {}'.format(hmmlearn.__version__))"
