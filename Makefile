.PHONY: format
format:
	black .

.PHONY: style
style:
	flake8 .

.PHONY: init
init:
	python3 -m MNIST.MNIST_main init