#!/bin/bash
train_batch_size=(500)
learning_rate_vals=(0.01)
epochs=(100)

for epoch_val in "${epochs[@]}"
do
	for lr_val in "${learning_rate_vals[@]}"
	do
		for batch_val in "${train_batch_size[@]}"
		do
			python3 CNN_TF_python.py $batch_val $lr_val $epoch_val
		done
	done
done