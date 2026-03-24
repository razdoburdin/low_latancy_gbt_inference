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
import argparse

import results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="xgboost")
    parser.add_argument("--n_instances", type=int)
    args = parser.parse_args()

    result = {}
    for instance_index in range(args.n_instances):
        instance_results = results.load_instance(args.framework, instance_index)
        
        for name, instance_result in instance_results.items():
            if not name in result:
                result[name] = []
            result[name] += instance_result["time"]

        # results.clear_instance(args.framework, instance_index)

    summary = {}    
    for name, times in result.items():
        mean = int(np.mean(times))
        p99 = int(np.percentile(times, 99))
        summary[name] = {"n_instances" : args.n_instances,
                         "total n_samples" : len(times),
                         "latancy [ns]" : {
                            "mean": mean,
                            "p99": p99
                         }
                        }

    results.save_summary(summary, args.framework, args.n_instances)