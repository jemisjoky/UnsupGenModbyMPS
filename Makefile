.PHONY: format
format:
	black .

.PHONY: style
style:
	flake8 .

.PHONY: init
init:
	python3 -m MNIST.MNIST_main init

.PHONY: rm-all-runs
rm-all-runs:
	rm -r MNIST/rand1k_runs/*
