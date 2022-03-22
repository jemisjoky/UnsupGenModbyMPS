.PHONY: format
format:
	black .

.PHONY: style
style:
	flake8 .

.PHONY: init
init:
	python3 -m train_script init

# .PHONY: train
# train:
# 	python3 -m train_script train_from_scratch

.PHONY: train-cluster
train-cluster:
	test -n "$(MSG)" # Must pass experiment message by setting MSG env variable
	./exp_tracker.py --gpus=1 --mem-per-cpu=16 train_script.py "$(MSG)" train_from_scratch

.PHONY: train-local
train-local:
	test -n "$(MSG)" # Must pass experiment message by setting MSG env variable
	./exp_tracker.py --gpus=0 --local train_script.py "$(MSG)" train_from_scratch

.PHONY: sample-cluster
sample-cluster:
	test -n "$(EXP_DIR)" # Must pass location of experiment directory
	sbatch --job-name=sample_mps --output=DELETE_ME.out --gpus=1 \
		--mem-per-cpu=32 --partition=unkillable \
		--export=ALL,EXP_DIR=$(EXP_DIR) sampler.py
.PHONY: continue
continue:
	python3 -m train_script continue

.PHONY: rm-all-runs
rm-all-runs:
	rm -r MNIST/rand1k_runs/*

.PHONY: rm-intermediate
rm-intermediate:
	python3 -m utils rm_intermediate
