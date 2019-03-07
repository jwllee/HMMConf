DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data

build:
	sudo docker build -t slac .

run: build
	sudo docker run -it -v $(DIR):/root/slac/code -v $(DATADIR):/root/slac/data slac /bin/bash
