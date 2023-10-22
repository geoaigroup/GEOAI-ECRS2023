# XAI Trainable Noise and Sobol 

[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository for our paper titled "Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).

## Overview
NoiseTraining-WHU and Sobol_WHU contain all the necessary files to reproduce the work done in the paper "Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation"

Inside the Folder Sobol_WHU:

- pytorch_example-WHU.ipynb is used to apply SOBOL as XAI method on the WHU building segmentation model. (sobol_attribution_method/torch_explainer.py could be editted to show intermediate results)


Inside the folder NoiseTraining-WHU:

- Run train.py to train the noise model over the WHU building segmentation model.
- Unoise_eval_method.ipynb shows results of the noise model.
- Unoise_eval_noise_count.ipynb compares between seg-grad-CAM and seg-grad-CAM++ over WHU building segmentation model.
- Unoise_eval_noise_count_newsobol.ipynb compares between SOBOL as XAI and seg-grad-CAM++.


