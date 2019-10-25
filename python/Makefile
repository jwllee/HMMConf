DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data

build:
	docker build -t hmmconf .

run: build
	docker run -it \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DIR):/home/hmmconf/code \
		-v $(DATADIR):/home/hmmconf/data \
		-e DISPLAY=unix$(DISPLAY) \
		--device /dev/snd:/dev/snd \
		hmmconf:latest /bin/bash
