# Novel Visual Category Discovery with Dual Ranking Statistics and Mutual Knowledge Distillation

## Dependencies

All dependencies are included in `requirements.txt`. To install, run

```shell
pip3 install -r requirements.txt
```

## Overview

We provide code for our experiments on CUB-200 and Stanford Cars.

## Data preparation

By default, we put the datasets in `/data/datasets/` and save trained models in `./data/experiments/` (soft link is suggested). 

- For CUB-200 dataset, download the dataset [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and put the unzipped data to `/data/dataset/cub200/`
- For Stanford-Cars dataset, download the dataset [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) and put the unzipped data to `/data/dataset/cars/`

Example training and testing split of CUB-200 and Stanford-Cars are presented in json format in the `asset` folder.
The format of the dataset json is as follows:

```json
{
    "path_type": "absolute",
    "nr_class": 200, // int, number of all classes,
    "info_dicts": [{
        "path": "/path/to/image.jpg",
        "label": 0,
    }, ...]
}
```

Due to the randomness of spliting the data, the final result should be an average of randomly split the dataset for several times.

## Self-supervised pretraining 

We use MoCoV2 with the ResNet50 backbone pretrained on ImageNet-1k for 800 epochs to initialze our model.
The pretrained model can be downloaded from the original repo [here](https://github.com/facebookresearch/moco).


## Novel category discovery on CUB200/Stanford-Cars


```shell
# Train and evaluation on CUB-200 
python3 ncd.py --custom_run cub --mode train --model_name resnet_fgvc --method gp --cls_num_from_json --moco_path /path/to/mocov2/ckpt --label_json_path_train asset/cub_novel_80_train.json --label_json_path_val asset/cub_novel_80_test.json --unlabel_json_path_train asset/cub_novel_20_train.json --unlabel_json_path_val asset/cub_novel_20_test.json

# Train and evaluation on Stanford-Cars
python3 ncd.py --custom_run cars --mode train --model_name resnet_fgvc --method gp --cls_num_from_json --moco_path /path/to/mocov2/ckpt --label_json_path_train asset/cars_novel_80_train.json --label_json_path_val asset/cars_novel_80_test.json --unlabel_json_path_train asset/cars_novel_20_train.json --unlabel_json_path_val asset/cars_novel_20_test.json
```


## Pretrained weight

Our pretrained weight for ResNet50 can be accessed from [here](https://drive.google.com/drive/folders/1MP_0qMKrjf1NI8NuQX55QgR5vTrR9O7r?usp=sharing)

## BibTex
If you use this code for your research, please cite our paper

```bibtex
@inproceedings{zhao21novel,
 author     = {Bingchen Zhao and Kai Han},
 title      = {Novel Visual Category Discovery with Dual Ranking Statistics and Mutual Knowledge Distillation},
 booktitle  = {Conference on Neural Information Processing Systems (NeurIPS)},
 year       = {2021}
}
```
