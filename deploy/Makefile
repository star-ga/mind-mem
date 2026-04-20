.PHONY: up down logs shell status build pull

COMPOSE := docker compose -f docker-compose.yml

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f mind-mem

shell:
	$(COMPOSE) exec mind-mem bash

status:
	$(COMPOSE) ps

build:
	$(COMPOSE) build --no-cache mind-mem

pull:
	$(COMPOSE) pull postgres ollama
