FROM python:3.7-stretch

RUN apt-get update && apt-get install -y \
	git \
	vim \
	make

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

WORKDIR /code
COPY . /code

