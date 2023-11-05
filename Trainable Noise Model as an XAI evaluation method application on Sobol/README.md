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
![sobol](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/65351288/9ed9d0b1-6e9c-449c-8f8c-20a9de6c03c0) | ![unoise](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/65351288/a4f7153f-ba0d-49fd-8c22-004b6dc7dad2)




## Getting Started

===> For all what follows, fix the paths relative to your system.

### Adapted Sobol application on WHU segmentation model

- Run Sobol_WHU/pytorch_example-WHU.ipynb. It applies the adapted sobol on a set of WHU dataset images. You con modify the code in SObol_WHU/sobol_attribution_method/torch_explainer.py to visualize intermediate results.

- The notebook saves the saliency maps produced by Adapted Sobol method, which you should copy and paste in NoiseTraining-WHU. It is needed when comparing between Adapted  Sobol and CAM++.

### Evaluating XAI methods with Noise model.

- NoiseTraining-WHU\train.py was used to train a noise model over the WHU building segmentation model.
- NoiseTraining-WHU\Unoise_eval_method.ipynb is used to inference the noise model and visualize the results of Unoise model as XAI method.
- NoiseTraining-WHU\Unoise_eval_noise_count.ipynb is used to compare between CAM and CAM++ using Unoise model as evaluation method.
- NoiseTraining-WHU\Unoise_eval_noise_count_sobol.ipynb is used to compare between Sobol and CAM++ using the Unoise model as evaluation method.


## Datasets

-[WHU Buildings dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 
