Steps to reproduce results:

## Install miniforge
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh -b
eval "$(miniforge3/bin/conda shell.bash hook)"
```

## Clone repo
```
git clone https://github.com/razdoburdin/low_latancy_gbt_inference.git
```

## Create env
```
cd low_latancy_gbt_inference
conda create --name gbt --file requirements.txt -y
conda activate gbt
```

## Convert xgboost model to treelite and onnx
```
python treelite_convert.py
python onnx_convert.py
```

## Launch benchmark wit 1, 8, 24, 48 instancies
```
./concurent_bench.sh 1
./concurent_bench.sh 8
./concurent_bench.sh 24
./concurent_bench.sh 48
```

## Look for results
```
cd summary
```
