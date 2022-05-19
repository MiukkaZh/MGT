# Learning Domain-Invariant Transformation for Speaker Verification
 
By Hanyi Zhang (<miukkazhang@gmail.com>), Longbiao Wang, Kong Aik Lee, Meng Liu, Jianwu Dang, Hui Chen.

**\[Todo\]**
- [x] Release the Cross-Genre trial, Cross-Dataset trial, and GCG benchmark.
- [x] Release the Meta Generalized Transformation (MGT) code.

## Introduction
This repository contains the PyTorch implementation for the paper _Learning Domain-Invariant Transformation for Speaker Verification_ in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2022). In this project, we provide Cross-Genre trial, Cross-Dataset trial, GCG benchmark, and Meta Generalized Transformation (MGT) code.

## Requirements
- Python >= 3.6
- Pytorch >= 1.2 and torchvision
- The `requirements.txt` file can be used to setup the environment.
```
conda create --name torch12 python=3.6
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

## Dataset:
You can download the `CN-Celeb` corpus here: [CN-Celeb](http://www.openslr.org/82/).

## Usage
### Feature Extraction
- Set `filelists/config.yaml` to the corresponding specific path.
- Extract features by running `filelists/Cnceleb/cnceleb_feature_extract.py`
```
python filelists/Cnceleb/cnceleb_feature_extract.py
```
- Build the mapping files (`cnceleb_test.json` and `cnceleb_train.json`) by running `filelists/database_gen.py`.
```
python filelists/database_gen.py
```
- Set `config.yaml` to the corresponding specific path.

### Train
- Run `meta_train.py` to train the model of Meta Generalized Transformation (MGT).
```
python meta_train.py --n_folder 0 --generalization gmlp --dataset cnceleb --domain g1,g2,g3,g4 --loss_type metaloss --n_shot 1 --n_query 2 --nb_class_train 100
```
### Evaluation
- Run `EER.py` to test the trained Meta Generalized Transformation (MGT) model.
```
python EER.py --model_type meta --data_type cnceleb --generalization gmlp --loss_type metaloss --dataset cnceleb --domain g1,g2,g3,g4 --veri_test_dir trials/cnceleb/fix_cnceleb_trials.txt --n_folder 15 --cp_num 0
```

## Citation
If you find the code useful for your research, please cite our paper.
```
@inproceedings{zhang2022learning,
  title={Learning Domain-Invariant Transformation for Speaker Verification},
  author={Zhang, Hanyi and Wang, Longbiao and Lee, Kong Aik and Liu, Meng and Dang, Jianwu and Chen, Hui},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7177--7181},
  year={2022},
  organization={IEEE}
}
```


## Note
- Two NVIDIA GeForce RTX 2080 Ti with 11G memory is adopted for training MGT.
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot), [meta-SR](https://github.com/seongmin-kye/meta-SR) and [g-mlp-pytorch](https://github.com/lucidrains/g-mlp-pytorch).