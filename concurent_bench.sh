#!/usr/bin/env bash

N_INSTANCES="$1"

for fw in xgboost daal4py treelite onnx; do
	for i in $(seq 0 $((N_INSTANCES - 1))); do
	    taskset -c "$i" \
        	python run_inference.py --framework=$fw --n_instances="$N_INSTANCES" --instance_index=${i} &
	done

	wait
	python collect.py --framework=$fw --n_instances="$N_INSTANCES"
done

