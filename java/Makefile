DIR=$(shell pwd)

build:
	docker build --tag=eclipse:2019-03 .

run: build
	mkdir -p .eclipse-docker
	mkdir -p workspace
	docker run -it --rm -d \
		-e DISPLAY=unix${DISPLAY} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(PWD)/.eclipse-docker:/home/developer \
		-v $(PWD)/workspace:/home/developer/workspace \
		eclipse:2019-03 

