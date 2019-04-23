FROM ubuntu:18.04
LABEL maintainer "Wai Lam Jonathan Lee <jonathan.wailam.lee@gmail.com>"
LABEL description="Dockerized eclipse"

RUN sed 's/main$/main universe/' -i /etc/apt/sources.list && \
    apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:webupd8team/java -y && \
    apt-get update && \
    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    apt-get install -y oracle-java8-installer libxext-dev libxrender-dev libxtst-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# Install libgtk as a separate step so that we can share the layer above with
# the netbeans image
ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	libgtk3.0 \
	libcanberra-gtk-module 

RUN wget http://eclipse.c3sl.ufpr.br/technology/epp/downloads/release/2019-03/R/eclipse-committers-2019-03-R-linux-gtk-x86_64.tar.gz -O /tmp/eclipse.tar.gz -q && \
    echo 'Installing eclipse' && \
    tar -xf /tmp/eclipse.tar.gz -C /opt && \
    rm /tmp/eclipse.tar.gz

RUN apt-get update && apt-get install -y \
	sudo

ADD run /usr/local/bin/eclipse

ENV HOME /home/developer
WORKDIR /home/developer

RUN useradd --home-dir $HOME developer \
	&& groupadd code \
	&& gpasswd -a developer code \
	&& chown -R developer:developer $HOME \
	&& chmod +x /usr/local/bin/eclipse

USER developer
CMD /usr/local/bin/eclipse
