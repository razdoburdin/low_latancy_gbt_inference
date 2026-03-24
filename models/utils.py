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

from pathlib import Path
import xgboost as xgb
import tl2cgen
import onnxruntime as ort
import json

def model_path(framework, name, ext=".json"):
    model_folder = "models"
    return f"{model_folder}/{framework}_{name}_model{ext}"

def get_model(name):
    path2model = model_path("xgboost", name)
    path = Path(path2model)
    if path.exists():
        booster = xgb.Booster()
        booster.load_model(path2model)

        config = json.loads(booster.save_config())
        return booster, config["learner"]["objective"]["name"]
    else:
        raise ValueError("Model file doesn't exist")

def get_treelite_model(name):
    libpath = model_path("treelite", name, ext=".so")
    path = Path(libpath)
    if path.exists():
        predictor = tl2cgen.Predictor(libpath)
        _, objective = get_model(name)
        return predictor, objective
    else:
        raise ValueError("Treelite model file doesn't exist. Run treelite_convert.py first.")

def get_onnx_model(name):
    onnx_path = model_path("onnx", name, ext=".onnx")
    path = Path(onnx_path)
    if path.exists():
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, sess_options)
        _, objective = get_model(name)
        return session, objective
    else:
        raise ValueError("ONNX model file doesn't exist. Run onnx_convert.py first.")
        