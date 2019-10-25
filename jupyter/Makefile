DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data
HMMCONF=$(shell dirname "$(shell pwd)")/python

build:
	docker build --tag=slac-jupyter .

jupyter: 
	docker run --rm -p 8888:8888 \
		-e JUPYTER_ENABLE_LAB=yes \
		-v $(DIR):/home/jovyan/work \
		-v $(DATADIR):/home/jovyan/data \
		-v $(HMMCONF):/home/jovyan/hmmconf \
		-v $(DIR)/../results:/home/jovyan/results \
		jupyter/datascience-notebook:9b06df75e445

bash: build
	docker run -it \
		-v $(DIR):/home/jovyan/work \
		-v $(DATADIR):/home/jovyan/data \
		-v $(HMMCONF):/home/jovyan/hmmconf \
		-v $(DIR)/../results:/home/jovyan/results \
		--network="host" \
		slac-jupyter /bin/bash
