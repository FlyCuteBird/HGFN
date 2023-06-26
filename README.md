# Heterogeneous Graph Fusion Network for Cross-modal Image-text Retrieval 


## Introduction
This is the source code of Heterogeneous Graph Fusion Network . It is built on top of the SCAN (https://github.com/kuanghuei/SCAN), GSMN in PyTorch.
We recommended the following dependencies.
* Python  2.7
* [PyTorch](http://pytorch.org/) 1.1.0
* numpy >= 1.16.6
* torchvision == 0.3.0
* pytorch-pretrained-bert == 0.6.2


## Pretrained results
If you don't want to train from scratch, you can download the pretrained results of HGFN from [here](https://drive.google.com/drive/folders/1b2Cx55i_ogAZsD9yXQtHlBUiMcVSAllK?usp=sharing), the results are reported in our paper. For the fusion of different pretrained results, it can be  be easily obtained from 'Fusion_Model.py'
```bash
Flickr30K; MSCOCO
HGFN-S:
Image to text: 75.3 93.9 97.7 || 78.7 95.1 98.3
Text to image: 57.8 83.0 89.3 || 62.5 89.7 95.5

HGFN-P:
Image to text: 75.3 94.2 97.2 || 76.4 95.2 98.2
Text to image: 57.4 83.1 89.6 || 62.8 90.3 95.6
```


## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN), and some other required data can be obtained from [here](https://drive.google.com/drive/folders/1UGmlc6noGFYoomPWXX13tIjvNzqOHTkq?usp=sharing) (for Flickr30K and MSCOCO) 

## Training

```bash
python train.py
```
## Evaluation
```bash
python test.py
```



