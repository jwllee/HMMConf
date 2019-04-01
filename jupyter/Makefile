DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data

build:
	sudo docker build --tag=slac-jupyter .

jupyter: 
	sudo docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v $(DIR):/home/jovyan/work -v $(DATADIR):/home/jovyan/data jupyter/datascience-notebook:9b06df75e445

bash: build
	sudo docker run -it -v $(DIR):/home/jovyan/work -v $(DATADIR):/home/jovyan/data slac-jupyter /bin/bash
