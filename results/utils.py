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
import json

instance_results_folder = "results/instances"
summary_results_folder = "summary"

def save_instance(result, framework, instance_index):
    path = Path(instance_results_folder)
    path.mkdir(exist_ok=True)

    file_name = f"{instance_results_folder}/{framework}_{instance_index}.json"

    with open(file_name, "w") as f:
        json.dump(result, f, indent=4)

def load_instance(framework, instance_index):
    file_name = f"{instance_results_folder}/{framework}_{instance_index}.json"

    with open(file_name, "r") as f:
        result = json.load(f)
    
    return result

def clear_instance(framework, instance_index):
    file_name = f"{instance_results_folder}/{framework}_{instance_index}.json"

    file_path = Path(file_name)
    if file_path.exists():
        file_path.unlink()


def save_summary(summary, framework, n_instances):
    path = Path(summary_results_folder)
    path.mkdir(exist_ok=True)
    file_name = f"{summary_results_folder}/{framework}_{n_instances}_instances_summary.json"

    with open(file_name, "w") as f:
        json.dump(summary, f, indent=4)