# LGGM

## Environment installation
Download anaconda/miniconda if needed

Create a rdkit environment that directly contains rdkit:
```
conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9
conda activate digress
```
Check that this line does not return an error:
```
python3 -c 'from rdkit import Chem'
```
Install graph-tool:
```
conda install -c conda-forge graph-tool=2.45
```
Check that this line does not return an error:
```
python3 -c 'import graph_tool as gt' 
```
Install the nvcc drivers for your cuda version. For example:
```
conda install -c "nvidia/label/cuda-11.8.0" cuda
```
Install a corresponding version of pytorch, for example:
```
pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
Install other packages using the requirement file:
```
pip install -r requirements.txt
```
Run:
```
pip install -e .
```
Navigate to the ./src/analysis/orca directory and compile orca.cpp:
```
g++ -O2 -std=c++11 -o orca orca.cpp
```

## Original Dataset
```
https://huggingface.co/datasets/YuWang0103/LGGM/tree/main
```


## Running
```
cd src
```




## Pre-training of the models 
```
iker_run_train_uniform_all_plus3.sh
iker_run_train_uniform_all.sh
```


## Fine-tuning
```
iker_run_train_ft_seed_uniform_erdos-renyi_plus3.sh
iker_run_train_ft_seed_uniform_erdos-renyi.sh
iker_run_train_ft_seed_uniform_InternetTopology_plus3.sh
iker_run_train_ft_seed_uniform_InternetTopology.sh
iker_run_train_ft_seed_uniform_USA-road_plus3.sh
iker_run_train_ft_seed_uniform_USA-road.sh
```


## Testing
```
iker_run_test_all_bare_2.sh
iker_run_test_all_bare.sh
iker_run_test_all_plus3_2.sh
iker_run_test_all_plus3.sh
iker_run_test_uniform_all.sh
iker_run_test_uniform_erdos-renyi.sh
iker_run_test_uniform_erdos-renyi.sh
iker_run_test_uniform_erdos-renyi.sh
iker_run_test_uniform_InternetTopology.sh
iker_run_test_uniform_USA-road.sh
```








Note that this repository is heavily built upon a public GitHub repository DiGress.
