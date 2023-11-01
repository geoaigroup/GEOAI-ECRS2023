# PEFTing TSViT 

[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository for our paper titled "Empirical Study of PEFT techniques for Winter Wheat Segmentation" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).

## Overview
Our research focuses on the use of Parameter Efficient Fine Tuning (PEFT) for the fine tune the TSViT model to the . This repository contains the code and resources used to reproduce the results presented in our paper. 


## Key Features

- This repository contains all the necessary python files and notebooks to reproduce the results of the paper on the Munich dataset (The Lebanese Dataset will be added on a later date, but all code required to run the experiments is available right now), specifically:
  - preprocessing done on the both the Lebanese and the Munich 480 dataset.
  - The PEFT techniques applied on TSViT model.
  - The training code for both Lebanon and Munich dataset.


## Some visual results





## Getting Started

Munich 480 dataset:
1. download the Munich dataset 
2. unzip and store data in the archive directory
3. run preporcess_munich.py
4. change the "run_config" from train_munich.py and run the file



## Datasets
Lebanese Dataset (Will be available later on)
Munich Dataset :https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480
