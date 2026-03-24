#!/usr/bin/env bash

N_INSTANCES="$1"

# for i in $(seq 0 $((N_INSTANCES - 1))); do
#     taskset -c "$i" \
#         python run_inference.py --framework=xgboost --n_instances="$N_INSTANCES" --instance_index=${i} &
# done

# wait
# python collect.py --framework=xgboost --n_instances="$N_INSTANCES"

for i in $(seq 0 $((N_INSTANCES - 1))); do
    taskset -c "$i" \
        python run_inference.py --framework=daal4py --n_instances="$N_INSTANCES" --instance_index=${i} &
done

wait
python collect.py --framework=daal4py --n_instances="$N_INSTANCES"

# for i in $(seq 0 $((N_INSTANCES - 1))); do
#     taskset -c "$i" \
#         python run_inference.py --framework=treelite --n_instances="$N_INSTANCES" --instance_index=${i} &
# done

# wait
# python collect.py --framework=treelite --n_instances="$N_INSTANCES"

# for i in $(seq 0 $((N_INSTANCES - 1))); do
#     taskset -c "$i" \
#         python run_inference.py --framework=onnx --n_instances="$N_INSTANCES" --instance_index=${i} &
# done

# wait
# python collect.py --framework=onnx --n_instances="$N_INSTANCES"