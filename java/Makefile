DIR=$(shell pwd)
JAVA_INSTALLER_FP=./jdk-installer.tar.gz

build:
	docker build --tag=eclipse:2019-03 --build-arg JAVA_INSTALLER_FP=$(JAVA_INSTALLER_FP) .

run: build
	mkdir -p .eclipse-docker
	mkdir -p workspace
	docker run -it --rm -d \
		-e DISPLAY=unix${DISPLAY} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(PWD)/../data:/home/developer/data \
		-v $(PWD)/.eclipse-docker:/home/developer \
		-v $(PWD)/workspace:/home/developer/workspace \
		-v $(PWD)/results:/home/developer/results \
		eclipse:2019-03 

