# Zero-Shot Refinement of Buildings Segmentation Models using SAM  

[![Conference](https://img.shields.io/badge/ECRS-Conference-brightgreen)](https://ecrs2023.sciforum.net/)
[![Workshop](https://img.shields.io/badge/NOAA%20Workshop-5th%20AI%20Demo-blue)](https://noaaai2023.sched.com/)

Welcome to the official repository for our paper titled "Zero-Shot Refinement of Buildings’ Segmentation Models using SAM" to appear at the 5th International Electronic Conference on Remote Sensing (ECRS 2023).
Paper: https://arxiv.org/abs/2310.01845
## Overview
![CNN_SAM_model](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/b2f6f42c-69ff-47b7-81c2-448d5c1fc85e)

Our research focuses on  a zero-shot refinement approach where the inference results of a CNN trained for buildings' segmentation from remote sensing images are passed as input to SAM. This repository contains the code and resources used to reproduce the results presented in our paper. Also, this repo hosts related materials for our live demo entitled "Zero-Shot Buildings' Segmentation using SAM" at the 5th NOAA Workshop on Leveraging AI in Environmental Sciences.




## Some visual results

Single Point          |          Skeleton Points          |          Bounding Box          
:-------------------------:|:-------------------------:|:-------------------------:
![2_550](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/06fd6fcf-c757-4fd6-aabc-7f7ddd28e97d) | ![2_54](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/51021610-b256-46e8-8083-4c1ab351a835) | ![2_52](https://github.com/geoaigroup/GEOAI-ECRS2023/assets/74465885/3427e084-0668-4641-bfb5-eb0fccb47529)   

## Installation 
Same packages and libraries as Segment anything model [SAM](https://github.com/facebookresearch/segment-anything) 

## Getting Started
Our code is designed to have predicted results of CNN model as shapefiles, and also it can be adjusted to have another input type.
First you can adjust paths of source images, predicted results of CNN model and output prediction in the path cell as illustrated:

```
# Paths
images = "data/images"
orig_shp="data/orig_shp"
skeleton_points="data/points.json"
cent_shp="data/center_points"
pred = "data/pred_shapefile"
output_dir = "data/MulticlassUnet_box_output"
score_dir = "data/MulticlassUnet_box_scores"
```

## Key Features

- AI models Implementation for building segmentation in remote sensing.
  - *Coming Soon!* Stay tuned!
  
- A comprehensive Colab notebook for a live demo presented at the 5th NOAA AI Workshop.
  - This notebook contains a demo of the implementation of two CNNs acting as prompt generators for [SAM](https://github.com/facebookresearch/segment-anything) on remote sensing data.
  - It also includes the use of [LangSAM](https://github.com/luca-medeiros/lang-segment-anything), an innovative integration of two powerful models: the foundation model [SAM](https://github.com/facebookresearch/segment-anything) and the visual grounding model [Grounding Dino]( https://github.com/IDEA-Research/GroundingDINO). This combination enables the use of textual prompts with SAM on remote sensing data.

## Citation:
```
@misc{mayladan2023zeroshot,
      title={Zero-Shot Refinement of Buildings' Segmentation Models using SAM}, 
      author={Ali Mayladan and Hasan Nasrallah and Hasan Moughnieh and Mustafa Shukor and Ali J. Ghandour},
      year={2023},
      eprint={2310.01845},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
