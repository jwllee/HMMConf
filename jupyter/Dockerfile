FROM jupyter/datascience-notebook:9b06df75e445
LABEL maintainer "Wai Lam Jonathan Lee <jonathan.wailam.lee@gmail.com>"

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root

RUN apt-get update && apt-get install -y \
	git \
	vim \
	make \
	build-essential
 
ENV HOME=/home/jovyan
WORKDIR $HOME

# setup dotfiles
RUN git clone --recursive https://github.com/jwllee/dotfiles.git \
	&& cd dotfiles && make 

# setup vim
RUN git clone --recursive https://github.com/jwllee/.vim.git .vim \
	&& ln -sf /.vim/vimrc /.vimrc \
	&& cd $HOME/.vim \
	&& git submodule update --init

RUN pip install PyQt5

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
