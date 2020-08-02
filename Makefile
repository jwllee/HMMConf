.DEFAULT_GOAL := help

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' ${MAKEFILE_LIST}

.PHONY: bash
bash: ## drop you into a running container
	@docker exec -it -e RUNTYPE=bash $$(docker ps|grep workspace|awk '{ print $$1 }') /docker-entrypoint.sh || true

.PHONY: rootbash
rootbash: ## drop you into a running container as root
	@docker exec -it -e RUNTYPE=bash --user=root $$(docker ps|grep workspace|awk '{ print $$1 }') /docker-entrypoint.sh || true

.PHONY: up
up: ## run the project
	@echo "Running default workspace..."
	@docker-compose run --service-ports --rm workspace || true

.PHONY: stop
stop: ## stop Docker containers without removing them
	@docker-compose stop

.PHONY: down
down: ## stop and remove Docker containers
	@docker-compose down --remove-orphans

.PHONY: pull
pull: ## update Docker images
	@docker-compose pull workspace jupyter
