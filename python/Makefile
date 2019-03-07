build:
	sudo docker build -t slac .

run: build
	sudo docker run -it -v "$$(pwd)":/code slac /bin/bash

