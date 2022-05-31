# Bag-of-Tricks-and-a-Strong-Baseline-for-Fungi-Fine-Grained-Classification



## 1. Environment setting 

### 1.0. Package
* Several important packages
    - torch == 1.10.2+cu111
    - trochvision == 0.11.3+cu111
    
* Replace folder timm/ to our timm/ folder (We made some changes to the original Timm framework, such as adding TA, etc)  
    
    #### pytorch model implementation [timm](https://github.com/rwightman/pytorch-image-models)

### 1.1. Dataset
In this project, we use a large fungi's datasets from this challenge to evaluate performance:
* [Fungi2022](https://www.kaggle.com/competitions/fungiclef2022/data)

### 1.2. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [x] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data
train data and test data structure:  
```
├── DF20/
│   ├── img20001.jpg
│   ├── img20002.jpg
│   └── ....
├── DF21/
│   ├── img21001.jpg
│   ├── img21002.jpg
│   └── ....
└──
```
  
Training sets and test sets are distributed with CSV labels corresponding to them.

### 2.2. configuration
you can directly modify yaml file (in ./configs/)

### 2.3. run.
```
python main.py --c ./configs/CUB200_SwinT.yaml
```
model will save in ./records/{project_name}/{exp_name}/backup/


### 2.4. about costom model
Building model refers to ./models/builder.py   
More detail in [how_to_build_pim_model.ipynb](./how_to_build_pim_model.ipynb)

### 2.5. multi-gpus
comment out main.py line 66
```
model = torch.nn.DataParallel(model, device_ids=None)
```  

## 3. Evaluation
For details, see test.sh
```
sh test.sh
```

### 3.1. please check yaml
set yaml (configuration file)
Key           | Value  | Description | 
--------------|:------|:------------| 
train_root    | ~      | set value to ~ (null) means this is not in training mode.  |
val_root  | ../data/eval/  |  path to validation samples |
pretrained  | ./pretrained/best.pt  |   pretrained model path |


../data/eval/ folder structure:  
```
├── eval/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```

### 3.2. run
```
python main.py --c ./configs/eval.yaml
```
results will show in terminal and been save in ./records/{project_name}/{exp_name}/eval_results.txt

## 4. HeatMap
```
python heat.py --pretrained ./best.pt --img ./imgs/001.jpg
```
![visualization](./imgs/test1_heat.jpg)

- - - - - - 

### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.
