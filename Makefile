.PHONY: format
format:
	black .

.PHONY: style
style:
	flake8 .

.PHONY: init
init:
	python3 -m MNIST.MNIST_main init

.PHONY: train-from-scratch
train-from-scratch:
	python3 -m MNIST.MNIST_main train_from_scratch

.PHONY: continue
continue:
	python3 -m MNIST.MNIST_main continue

.PHONY: rm-all-runs
rm-all-runs:
	rm -r MNIST/rand1k_runs/*

.PHONY: rm-intermediate
rm-intermediate:
	python3 -m utils rm_intermediate
