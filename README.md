# GEOAI-ECRS2023 Papers  

[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository of GEOAI papers published in the 5th International Electronic Conference on Remote Sensing (ECRS 2023). Here you can explore a comprehensive collection of insightful research papers covering various topics in the field of remote sensing. We invite you to delve into the latest advancements and discoveries presented by our esteemed researchers and authors.

**Table Of Contents**
* [Empirical Study of PEFT techniques for Winter Wheat Segmentation](#empirical-study-of-peft-techniques-for-winter-wheat-segmentation)
* [Zero-Shot Refinement of Buildings’ Segmentation Models using SAM](#zero-shot-refinement-ofbuildings’-segmentation-models-using-SAM)
* [Extending GradCam-based XAI methods for Remote Sensing Imagery Segmentation](#extending-GradCam-based-XAI-methods-for-remote-sensing-imagery-segmentation)
* [Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation](#trainable-noise-model-as-an-XAI-evaluation-method-application-on-Sobol-for-remote-sensing-image-segmentation)


## [Empirical Study of PEFT techniques for Winter Wheat Segmentation]

![image](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/8a6569e8-a987-4457-8d5c-3ee6a5d34a72)

The aim of this work is to explore efficient fine-tuning approaches for crop monitoring. Specifically, we focus on adapting the SOTA TSViT model, recently proposed in CVPR 2023, to address winter wheat field segmentation, a critical task for crop monitoring and food security, especially following the Ukrainian conflict, given the economic importance of wheat as a staple and cash crop in various regions. 

### Citation

```
M. Zahweh, H. Nasrallah, M. Shukor, G. Faour and A. J. Ghandour, “Empirical Study of PEFT techniques for Winter Wheat Segmentation”, in 5th International Electronic Conference on Remote Sensing, Nov 17 - Nov 21, 2023.  
```

## [Zero-Shot Refinement of Buildings’ Segmentation Models using SAM](https://geogroup.ai/publication/2023ecrs_zeroshotsam/2023ECRS_ZeroShotSAM.pdf)

![CNN_SAM_model](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/ef2940ca-2998-43a5-943b-2dbc4461004f)

We introduce different prompting strategies, including integrating a pre-trained CNN as a prompt generator. This novel approach augments SAM with recognition abilities, a first of its kind. We evaluated our method on three remote sensing datasets, including the WHU Buildings dataset, the Massachusetts Buildings dataset, and the AICrowd Mapping Challenge.

### Citation
```
A. Mayladan, H. Nasrallah, H. Moughnieh, M. Shukor and A. J. Ghandour, “Zero-Shot Refinement of Buildings’ Segmentation Models using SAM”, in 5th International Electronic Conference on Remote Sensing, Nov 17 - Nov 21, 2023.  
```

## [Extending GradCam-based XAI methods for Remote Sensing Imagery Segmentation](https://geogroup.ai/publication/2023ecrs_camentropy/2023ECRS_CAMEntropy.pdf)

![image](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/3e224412-9ac5-47f9-8917-95fc74ac5ab6)

This paper offers to bridge this gap by adapting the recent XAI classification algorithms and making them usable for muti-class image segmentation, where we mainly focus on buildings’ segmentation from high-resolution satellite images. To benchmark and compare the performance of the proposed approaches, we introduce a new XAI evaluation methodology and metric based on "Entropy" to measure the model uncertainty.

### Citation
```
A. GIZZINI, M. Shukor and A. J. Ghandour, “Extending GradCam-based XAI methods for Remote Sensing Imagery Segmentation”, in 5th International Electronic Conference on Remote Sensing, Nov 17 - Nov 21, 2023.  
```

## [Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation](https://geogroup.ai/publication/2023ecrs_noisesobol/2023ECRS_NoiseSobol.pdf)
![image](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/8a0dca1a-989d-4c44-9bab-edb742d0b51a)

This paper adapts the recent gradient-free Sobol XAI method for semantic segmentation. To measure the performance of the Sobol method for segmentation, we propose a quantitative XAI evaluation method based on a learnable noise model. The main objective of this model is to induce noise on the explanation maps, where higher induced noise signifies low accuracy and vice versa. 

### Citation
```
H. Shreim, A. GIZZINI and A. J. Ghandour, “Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation”, in 5th International Electronic Conference on Remote Sensing, Nov 17 - Nov 21, 2023.  
```

