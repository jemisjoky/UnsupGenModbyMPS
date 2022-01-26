.PHONY: format
format:
	black .

.PHONY: style
style:
	flake8 .

.PHONY: train
train:
	python3 -m continuous-umps.train_scripts
