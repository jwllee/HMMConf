DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data

build:
	docker build -t lac .

run: build
	docker run -it \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DIR):/root/slac/code \
		-v $(DATADIR):/root/slac/data \
		-e DISPLAY=unix$(DISPLAY) \
		--device /dev/snd:/dev/snd \
		lac /bin/bash
