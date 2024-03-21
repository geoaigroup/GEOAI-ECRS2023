# Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation


[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)

Welcome to the official repository for our paper titled "Extending GradCam-based XAI methods for Remote Sensing Imagery Segmentation" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).
Paper: [arXiv](https://arxiv.org/pdf/2310.01837.pdf)

## Overview

This paper offers to bridge this gap by adapting the recent XAI classification algorithms and making them usable for muti-class image segmentation, where we mainly focus on buildings’ segmentation from high-resolution satellite images. To benchmark and compare the performance of the proposed approaches, we introduce a new XAI evaluation methodology and metric based on "Entropy" to measure the model uncertainty.

## Some visual results

![276731622-3e224412-9ac5-47f9-8917-95fc74ac5ab6](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/78584545/d87c1af5-c7fc-4fc9-8d5b-4d97dbe9756e)




## Installation 

In order to properly run the paper's code, the following packages should be installed:
1. grad-cam
2. segmentation_models_pytorch 
3. numpy
4. opencv-python
5. torch
6. skimage.io (scikit-image)
7. rasterio
8. imageio
9. matplotlib
10. ttach 
11. tqdm

Follow these steps:

1. **Download CUDA 12**: Make sure you have CUDA 12 installed on your system. You can download it from the official NVIDIA website.

2. **Create a Virtual Environment**: Create a virtual environment using Python's venv module. Run the following command:

    ```bash
    $ python -m venv .venv/eo-xai
    ```
    
3. **Activate the Virtual Environment**: Activate the virtual environment by running the following command:

    ```bash
    $ source .venv/eo-xai/bin/activate
    ```

4. **Install Project Requirements**: Install the necessary libraries by navigating to the project directory and running:

    ```bash
    $ cd path/to/your/project
    $ pip install -r requirements.txt
    ```

Make sure to replace "path/to/your/project" with the actual path to your project directory containing the requirements.txt file.

## Implementation
In order to reproduce the reported results, you need simply to go through the grad_cam_extensions.ipynb notebook that executes the following steps in order:

1. Cell 1: Import the required packages
2. Cell 2: Load the pre-trained model weights
3. Cell 3: Define the CAM-based Extensions
4. Cell 4: Apply the adapted CAM-based Extensions to the considered dataset. Here you should define the following parameters:
   - XAI_method: define the studied cam-based methods. Full list = ["grad_cam", "hires_cam", "ew_cam", "grad_cam_pp", "x_grad_cam","score_cam", "layer_cam", "eigen_cam", "eigen_grad_cam"]
   - target_layers: define the considered target layer. In the code, we considered the first decoder block, where target_layers =  [model.decoder.blocks[decoder_idx - 1]]
   - target_category: define the target class to be interpreted. In the code, we consider target_category = 0 which corresponds to the building class.

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

