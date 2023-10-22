# XAI Trainable Noise and Sobol 

[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository for our paper titled "Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).
Paper: [arXiv](https://arxiv.org/abs/2310.01828)

## Overview
Our research focuses on XAI methods for segmentation models and the evaluation techniques for existing XAI methods. This repository contains the code and resources used to reproduce the results presented in our paper. 


## Key Features

- This repository contains all the necessary python files and notebooks to reproduce the results of the paper, specifically:
  - Adapting Sobol XAI method for segmentation tasks and testing it on WHU building segmentation model.
  - Training Noise model on the WHU building segmentation model and using it as evaluation method for existing XAI methods like Seg-grad-CAM, seg-grad-CAM++ and our adapted sobol method.



## Some visual results

Adapted Sobol         |          Unoise model              
:-------------------------:|:-------------------------:



## Getting Started


Inside the Folder Sobol_WHU:

- pytorch_example-WHU.ipynb is used to apply SOBOL as XAI method on the WHU building segmentation model. (sobol_attribution_method/torch_explainer.py could be editted to show intermediate results)


Inside the folder NoiseTraining-WHU:

- Run train.py to train the noise model over the WHU building segmentation model.
- Unoise_eval_method.ipynb shows results of the noise model.
- Unoise_eval_noise_count.ipynb compares between seg-grad-CAM and seg-grad-CAM++ over WHU building segmentation model.
- Unoise_eval_noise_count_newsobol.ipynb compares between SOBOL as XAI and seg-grad-CAM++.


## Datasets

-[WHU Buildings dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 