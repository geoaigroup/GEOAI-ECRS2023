# Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation


[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository for our paper titled "Extending GradCam-based XAI methods for Remote Sensing Imagery Segmentation" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).
Paper: [arXiv](https://arxiv.org/pdf/2310.01837.pdf)

## Overview

This paper offers to bridge this gap by adapting the recent XAI classification algorithms and making them usable for muti-class image segmentation, where we mainly focus on buildings’ segmentation from high-resolution satellite images. To benchmark and compare the performance of the proposed approaches, we introduce a new XAI evaluation methodology and metric based on "Entropy" to measure the model uncertainty.

## Some visual results

![276731622-3e224412-9ac5-47f9-8917-95fc74ac5ab6](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/78584545/d87c1af5-c7fc-4fc9-8d5b-4d97dbe9756e)


## Installation 


## Implementation
To run our approach prediction you need to run main function and you can change arguments in order to select specific CNN model and select specific prompt type:

For the CNN model choose either *multiclassUnet* (Multiclass Unet CNN model) or *DCNN* (Dlink-Net CNN model).

For prompt type you can choose one of these prompts 


## Datasets

-[WHU Buildings dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 



## Citation:

```
@misc{gizzini2023extending,
      title={Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation}, 
      author={Abdul Karim Gizzini and Mustafa Shukor and Ali J. Ghandour},
      year={2023},
      eprint={2310.01837},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

