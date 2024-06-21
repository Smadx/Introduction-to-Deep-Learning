# Competition:GoodBooks RecSys
## File
You should organize files like this:
```
---Competition
 |
 |---dataset
 | |---train_data.csv
 | |---val_data.csv
 | |---train_dataset.csv
 | |---test_dataset.csv
 | |---submission.csv
 |
 |---results
 |---src
 |---data.ipynb
 |---Readme.md
 |---environment.yml
```
## Environment
We use conda to manage the environment.You can create the same environment with conda:
```shell
conda env create -f environment.yml
```
## Method
We build our RedSys referenced from [LightGCN](https://arxiv.org/pdf/2002.02126.pdf)
You can train the model with default config by: 
**Linux**
```shell
cd src
bash train.sh
```
**Windows**
```shell
cd src
accelerate launch --config_file accelerate_config.yaml train.py --results_path "results/train1/"
```