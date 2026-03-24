# ===============================================================================
# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import numpy as np
import time
import argparse
import daal4py
import tl2cgen
import onnxruntime as ort
import os
import glob
import shutil
from pathlib import Path

import datasets
import models
import results

def process_xgb_prediction(objective, prediction):
    if (objective == 'binary:logistic'):
        prob = prediction
        prediction = (prob >= 0.5).astype(int)
    elif (objective == 'multi:softprob'):
        prob = prediction
        prediction = np.array([np.argmax(prob[i]) for i in range(prob.shape[0])])
    return prediction
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="xgboost")
    parser.add_argument("--n_instances", type=int, default=1)
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=4096)
    args = parser.parse_args()

    barrier_dir = "bench_barrier"
    os.makedirs(barrier_dir, exist_ok=True)

    instance_index = args.instance_index
    n_instances = args.n_instances
    max_samples = args.max_samples
    result = {}
    for name in datasets.dataset_loaders:
        [X, y] = datasets.get_data(name, "test")
        block_size = int(np.floor(X.shape[0] / n_instances))

        begin = block_size * instance_index
        end = np.minimum(block_size * (instance_index + 1), X.shape[0])
        n_samples = np.minimum(max_samples, end - begin)

        booster, objective = models.get_model(name)
        result[name] = {}
        result[name]["time"] = np.zeros(n_samples, dtype=int)

        if "multi:" in objective:
            num_class = len(np.unique(y))
            prediction = np.zeros(shape=(n_samples, num_class))
        else:
            prediction = np.zeros(n_samples)

        if args.framework == "treelite":
            X = np.asarray(X, np.float32)
            predictor, objective = models.get_treelite_model(name)
        if args.framework == "onnx":
            X = np.asarray(X, np.float32)
            session, objective = models.get_onnx_model(name)
            input_name = session.get_inputs()[0].name

        # Signal this instance is ready
        open(f"{barrier_dir}/{name}_{args.framework}_{instance_index}.ready", "w").close()
        # Wait for all instances
        while len(glob.glob(f"{barrier_dir}/{name}_{args.framework}_*.ready")) < n_instances:
            pass  # spin-wait

        if args.framework == "xgboost":
            # warm-up
            for sample in range(64):
                _ = booster.inplace_predict(X[sample:sample+1, :])
            for idx in range(n_samples):
                sample = block_size * instance_index + idx
                begin = time.perf_counter_ns()
                out = process_xgb_prediction(objective, booster.inplace_predict(X[sample:sample+1, :]))
                end = time.perf_counter_ns()
                prediction[idx:idx+1] = out
                result[name]["time"][idx] = end - begin

        elif args.framework == "daal4py":
            X = np.asarray(X, np.float32)
            model_daal = daal4py.mb.convert_model(booster)
            # warm-up
            for sample in range(64):
                _ = model_daal.predict(X[sample:sample+1, :])
            for idx in range(n_samples):
                sample = block_size * instance_index + idx
                begin = time.perf_counter_ns()
                out = model_daal.predict(X[sample:sample+1, :])
                end = time.perf_counter_ns()
                prediction[idx:idx+1] = out
                result[name]["time"][idx] = end - begin

        elif args.framework == "treelite":
            # warm-up
            for sample in range(64):
                _ = predictor.predict(tl2cgen.DMatrix(X[sample:sample+1, :]))
            for idx in range(n_samples):
                sample = block_size * instance_index + idx
                begin = time.perf_counter_ns()
                out = predictor.predict(tl2cgen.DMatrix(X[sample:sample+1, :]))
                end = time.perf_counter_ns()
                prediction[idx:idx+1] = out
                result[name]["time"][idx] = end - begin

        elif args.framework == "onnx":
            # warm-up
            for sample in range(64):
                _ = session.run(None, {input_name: X[sample:sample+1, :]})
            for idx in range(n_samples):
                sample = block_size * instance_index + idx
                begin = time.perf_counter_ns()
                out = session.run(None, {input_name: X[sample:sample+1, :]})
                end = time.perf_counter_ns()
                prediction[idx:idx+1] = out[0]
                result[name]["time"][idx] = end - begin

        else:
            raise ValueError("Unknown framework")

        result[name]["time"] = result[name]["time"].tolist()

    results.save_instance(result, args.framework, instance_index)

    if instance_index == 0:
        folder = Path(barrier_dir)
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)

