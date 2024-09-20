# Python code for "Deep Learning Based Superposition Coded Modulation for Hierarchical Semantic Communications over Broadcast Channels"
This repository contains the original code and models for the work _Deep Learning Based Superposition Coded Modulation for Hierarchical Semantic Communications over Broadcast Channels_[1].

[1] Y. Bo, S. Shao and M. Tao, "Deep Learning Based Superposition Coded Modulation for Hierarchical Semantic Communications over Broadcast Channels," in IEEE Transactions on Communications, doi: 10.1109/TCOMM.2024.3447870.

## Requirements
* matplotlib==3.7.5
* numpy==1.24.4
* numpy==1.24.3
* pandas==2.0.3
* scikit-image==0.21.0
* torch==2.3.1
* torchvision==0.18.1
* tqdm==4.66.4

## Training & Evaluation
This code implements the superposition of 4qam and 4qam, as well as the superposition for 4qam and 16qam.
'./models' contains some trained model parameters as examples. (To be uploaded)

For training, first train the pretrain model then the SCM model.
To train the pretrain model, run the following command (as an example):
```
python main.py --net 'analog_good' --mode 'train' --order 16
```
To train the SCM model, run the following command (as an example):
```
python main.py --net 'scm' --mode 'train' --sp_mode '4and4' --a 0.8
```
It's perfectly OK not to train or load the pretrain model, just comment out the corresponding lines.

For evaluation, run the following command (as an example):
```
python main.py --net 'scm' --mode 'test' --sp_mode '4and4' --a 0.8
```
