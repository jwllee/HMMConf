DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data

build:
	sudo docker build -t slac .

run: build
	sudo docker run -it \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DIR):/root/slac/code \
		-v $(DATADIR):/root/slac/data \
		-e DISPLAY=unix$(DISPLAY) \
		--device /dev/snd:/dev/snd \
		slac /bin/bash
