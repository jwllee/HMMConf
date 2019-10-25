FROM ubuntu:18.04
LABEL maintainer "Wai Lam Jonathan Lee <jonathan.wailam.lee@gmail.com>"
LABEL description="Dockerized eclipse"

ARG JAVA_INSTALLER_FP

# Install libgtk as a separate step so that we can share the layer above with
# the netbeans image
ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	libgtk3.0 \
	libcanberra-gtk-module \
	wget

RUN wget http://eclipse.c3sl.ufpr.br/technology/epp/downloads/release/2019-03/R/eclipse-committers-2019-03-R-linux-gtk-x86_64.tar.gz -O /tmp/eclipse.tar.gz -q && \
    echo 'Installing eclipse' && \
    tar -xf /tmp/eclipse.tar.gz -C /opt && \
    rm /tmp/eclipse.tar.gz

RUN apt-get update && apt-get install -y \
	sudo

ENV HOME /home/developer
WORKDIR /home/developer

COPY $JAVA_INSTALLER_FP $HOME/
RUN echo $(which update-alternatives)
RUN tar -xvf /home/developer/jdk-installer.tar.gz \
	&& mkdir /usr/lib/jvm \
	&& mv ./jdk1.8.0_221 /usr/lib/jvm \
	&& echo $(ls /usr/lib/jvm)
	
RUN update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk1.8.0_221/bin/java" 1 \
	&& update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/jdk1.8.0_221/bin/javac" 1 \
    && update-alternatives --install "/usr/bin/javaws" "javaws" "/usr/lib/jvm/jdk1.8.0_221/bin/javaws" 1

RUN chmod a+x /usr/bin/java \
	&& chmod a+x /usr/bin/javac \
	&& chmod a+x /usr/bin/javaws \
	&& chown -R root:root /usr/lib/jvm/jdk1.8.0_221

RUN echo $(java -version)

ADD run /usr/local/bin/eclipse

RUN useradd --home-dir $HOME developer \
	&& groupadd code \
	&& gpasswd -a developer code \
	&& chown -R developer:developer $HOME \
	&& chmod +x /usr/local/bin/eclipse

USER developer
CMD /usr/local/bin/eclipse
