PROJECT_NAME := openai-access-gateway

# VERSION is the version we should download and use.
VERSION:=$(shell git rev-parse --short HEAD)
# DOCKER is the docker image repo we need to push to.
DOCKER_REPO:=defangio
DOCKER_IMAGE_NAME:=$(DOCKER_REPO)/$(PROJECT_NAME)

DOCKER_IMAGE_ARM64:=$(DOCKER_IMAGE_NAME):arm64-$(VERSION)
DOCKER_IMAGE_AMD64:=$(DOCKER_IMAGE_NAME):amd64-$(VERSION)

.PHONY: image-amd64
image-amd64:
	docker build --platform linux/amd64 -f ./src/Dockerfile_ecs -t ${PROJECT_NAME} -t $(DOCKER_IMAGE_AMD64) --provenance false ./src

.PHONY: image-arm64
image-arm64:
	docker build --platform linux/arm64 -f ./src/Dockerfile_ecs -t ${PROJECT_NAME} -t $(DOCKER_IMAGE_ARM64) --provenance false ./src

.PHONY: images
images: image-amd64 image-arm64 ## Build all docker images and manifest

.PHONY: push-images
push-images: images login ## Push all docker images
	docker push $(DOCKER_IMAGE_AMD64)
	docker push $(DOCKER_IMAGE_ARM64)
	docker manifest create --amend $(DOCKER_IMAGE_NAME):$(VERSION) $(DOCKER_IMAGE_AMD64) $(DOCKER_IMAGE_ARM64)
	docker manifest push --purge $(DOCKER_IMAGE_NAME):$(VERSION)

.PHONY: no-diff
no-diff:
	git diff-index --quiet HEAD --       # check that there are no uncommitted changes

.PHONY: push
push: no-diff push-images ## Push all docker images and "latest" manifest
	docker manifest create --amend $(DOCKER_IMAGE_NAME):latest $(DOCKER_IMAGE_AMD64) $(DOCKER_IMAGE_ARM64)
	docker manifest push --purge $(DOCKER_IMAGE_NAME):latest

.PHONY: login
login: ## Login to docker
	@docker login
