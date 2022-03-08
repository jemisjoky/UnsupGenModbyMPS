.PHONY: format
format:
	black .

.PHONY: style
style:
	flake8 .

.PHONY: init
init:
	python3 -m MNIST.MNIST_main init

.PHONY: train
train:
	python3 -m MNIST.MNIST_main train_from_scratch

.PHONY: train-cluster
train-cluster:
	test -n "$(MSG)" # Must pass experiment message by setting MSG env variable
	# ./exp_tracker.py --gpus=1 MNIST/MNIST_main.py "$(MSG)" train_from_scratch
	./exp_tracker.py --gpus=0 MNIST/MNIST_main.py "$(MSG)" train_from_scratch

.PHONY: continue
continue:
	python3 -m MNIST.MNIST_main continue

.PHONY: rm-all-runs
rm-all-runs:
	rm -r MNIST/rand1k_runs/*

.PHONY: rm-intermediate
rm-intermediate:
	python3 -m utils rm_intermediate
