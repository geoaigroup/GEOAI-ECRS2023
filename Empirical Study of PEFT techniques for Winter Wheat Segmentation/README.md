# PEFTing TSViT 

[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository for our paper titled "Empirical Study of PEFT techniques for Winter Wheat Segmentation" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).  
Paper: [arXiv](https://arxiv.org/pdf/2310.01825v1.pdf)

## Overview
Our research focuses on the use of Parameter Efficient Fine Tuning (PEFT) for the fine tune the TSViT model. This repository contains the code and resources used to reproduce the results presented in our paper. 


## Key Features

- This repository contains all the necessary python files and notebooks to reproduce the results of the paper on the Munich dataset (The Lebanese Dataset will be added on a later date, but all code required to run the experiments is available right now), specifically:
  - preprocessing done on the both the Lebanese and the Munich 480 dataset.
  - The PEFT techniques applied on TSViT model.
  - The training code for both Lebanon and Munich dataset.

## Getting Started

* <u>Beqaa-Lebanon dataset</u>:
1. Dataset [Docuemntation](https://github.com/user-attachments/files/16637950/Lebanese.dataset.documentation-itr1.pdf))
2. Reach out to us if you need access to this dataset.
  
* <u>Munich 480 dataset</u>:
1. [Download](https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480) the Munich dataset 
2. Unzip the file in "Munich480" directory
3. Run preporcess_munich.py
4. change the "run_config" from train_munich.py or run.py and run the file


## Some visual results
* <u>Beqaa-Lebanon dataset</u>:
![image](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/8a6569e8-a987-4457-8d5c-3ee6a5d34a72)
<!-- ![app-gui](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/14883982/bc918eea-5afa-4bf7-9323-90aeea12d393) -->

* <u>Munich 480 dataset</u>:
  
![image](https://github.com/geoaigroup/GEOAI-ECRS2023/blob/main/Empirical%20Study%20of%20PEFT%20techniques%20for%20Winter%20Wheat%20Segmentation/images/Munich%20Results.png)


## Contact
Feel free to reach to us via below link or through aghandour at cnrs.edu.lb:
https://geogroup.ai/#contact

