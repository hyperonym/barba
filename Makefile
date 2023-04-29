NAME := barba
VERSION := 0.2.0

DOCKER_HUB_OWNER ?= hyperonym
DOCKER_HUB_IMAGE := $(DOCKER_HUB_OWNER)/$(NAME):$(VERSION)

GITHUB_PACKAGES_OWNER ?= hyperonym
GITHUB_PACKAGES_IMAGE := ghcr.io/$(GITHUB_PACKAGES_OWNER)/$(NAME):$(VERSION)

TARGET_CONTAINER_PLATFORMS := linux/amd64

.PHONY: build
build:
	@python convert.py hyperonym/barba models/barba
	@python export.py

.PHONY: changelog
changelog:
	@mkdir -p dist
	@git log $(shell git describe --tags --abbrev=0 2> /dev/null)..HEAD --pretty='tformat:* [%h] %s' > dist/changelog.md
	@cat dist/changelog.md

.PHONY: clean
clean:
	@find . -type f -name "*.py[co]" -delete && find . -type d -name "__pycache__" -delete
	@rm -rf .coverage .pytest_cache/ *.egg-info/ build/ dist/ htmlcov/ servables/

.PHONY: docker
docker:
	@docker build --tag $(DOCKER_HUB_IMAGE) .

.PHONY: docker-hub
docker-hub:
	@docker buildx build --push --platform $(TARGET_CONTAINER_PLATFORMS) --tag $(DOCKER_HUB_IMAGE) .

.PHONY: github-packages
github-packages:
	@docker buildx build --push --platform $(TARGET_CONTAINER_PLATFORMS) --tag $(GITHUB_PACKAGES_IMAGE) .

.PHONY: github-release
github-release: changelog
	@gh release create v$(VERSION) -F dist/changelog.md -t v$(VERSION)

.PHONY: install
install:
	@pip install -U pip
	@pip install -r requirements.txt
