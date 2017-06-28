#!/bin/bash
#learning_rate_vals=(2 3 4 5)
#epochs=(36 24 20 36)
declare -A pairs=( [5]=36) # [3]=24 [4]=20)
for i in "${!pairs[@]}"; do
  j=${pairs[$i]}
  python3 train_cnn.py $i $j
done