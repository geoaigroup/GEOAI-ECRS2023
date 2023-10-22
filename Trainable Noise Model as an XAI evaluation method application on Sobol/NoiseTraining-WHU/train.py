import albumentations as A
import cv2
from glob import glob
import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_cityscapes import *
from epoch import *
import neptune
import pickle
from segmentation_models_pytorch import Unet
import torch
import argparse
import os
import cv2
from segmentation_models_pytorch import Unet
from skimage.io import imread,imsave
import numpy as np
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import imageio
from utils import *
from rasterio.features import shapes
from tqdm.notebook import tqdm


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# ======== CONFIGURATION ======== #

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

S_EXPERIMENT = "UNET_EfficientNetB4_CE"
P_DIR_CKPT = os.path.join("/home/jamada/jupyterlab/eo-xai/whu_noise_training/", S_EXPERIMENT, "Checkpoints")
P_DIR_LOGS = os.path.join("/home/jamada/jupyterlab/eo-xai/whu_noise_training/", S_EXPERIMENT, "Logs")
P_DIR_EXPORT = os.path.join("/home/jamada/jupyterlab/eo-xai/whu_noise_training/", S_EXPERIMENT, "Export")
S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S_NAME_ENCODER = "efficientnet-b4"
S_NAME_WEIGHTS = "imagenet"
N_EPOCH_MAX = 50
N_SIZE_BATCH_TRAINING = 4  # training batch size
N_SIZE_BATCH_VALIDATION = 4  # validation batch size
N_SIZE_BATCH_TEST = 4  # test batch size
N_STEP_LOG = 3  # evaluate on validation set and save model every N iterations
N_WORKERS = 8  # to be adapted for each system
P_DIR_MODEL = '/home/jamada/UrbanModels/weights_pytorch/tu-tf_efficientnet_b0_Unet_whu_3-classes_40-epochs_TRY001'



#========= initialize neptune ========== #
neptune_run = neptune.init_run(
    project="GEOgroup/eo-xai", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNjUxZmRmOC0wZWMyLTQ3M2QtODdhNS03ZDY2NDBjODRhNmIifQ==",
)  


# ======== LOAD UTILITY MODEL ======== #
#load your model with pre-trained weights
utility_model = Unet(
        encoder_name = "tu-tf_efficientnet_b0",
        encoder_depth= 5,
        encoder_weights = None,
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32, 16),
        decoder_attention_type = None,
        in_channels= 3,
        classes = 3,
        activation = 'sigmoid',
        aux_params = None,
    )

utility_model = load_model(utility_model,P_DIR_MODEL)
utility_model.cuda()

# ======== SETUP ======== #
# U-Net model
model = smp.Unet(
   encoder_name = S_NAME_ENCODER,
   encoder_weights = S_NAME_WEIGHTS,
   in_channels = 3,
   classes = 1,
)


# setup input normalization
preprocess_input = get_preprocessing_fn(
    encoder_name = S_NAME_ENCODER,
    pretrained = S_NAME_WEIGHTS,
)

transform_full = A.Compose([
    # A.Resize(512,512),
    # A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])

# setup datasets
dataset_training = DataLoaderSegmentation(
    image_path = "/home/jamada/jupyterlab/datasets/WHU/WHU_building/train/A",
    mask_path = "/home/jamada/jupyterlab/datasets/WHU/WHU_building/train/OUT",
    transform = transform_full,
    device = S_DEVICE
)
dataset_validation = DataLoaderSegmentation(
    image_path = "/home/jamada/jupyterlab/datasets/WHU/WHU_building/val/A",
    mask_path = "/home/jamada/jupyterlab/datasets/WHU/WHU_building/val/OUT",
    transform = transform_full,
    device = S_DEVICE
)
print("dataset_training len", len(dataset_training))
print("dataset_validation len", len(dataset_validation))


# setup data loaders
loader_training = DataLoader(
    dataset_training,
    batch_size = N_SIZE_BATCH_TRAINING,
    shuffle = True,
    num_workers = N_WORKERS,
)
loader_validation = DataLoader(
    dataset_validation,
    batch_size = N_SIZE_BATCH_VALIDATION,
    shuffle = False,
    num_workers = N_WORKERS,
)


# setup loss
loss = torch.nn.BCELoss()
loss.__name__ = "total_loss"

# setup optimizer
LR_max = 5e-4
LR_min = 1e-6
optimizer = torch.optim.Adam([
    dict(params = model.parameters(), lr =LR_max),
])
# setup learning rate scheduler
# (here cosine decay that reaches 1e-6 after N_EPOCH_MAX)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer = optimizer,
    T_max = N_EPOCH_MAX,
    eta_min = LR_min,
)
# setup Tensorboard logs writer
os.makedirs(os.path.join(P_DIR_LOGS, "Training"), exist_ok = True)
os.makedirs(os.path.join(P_DIR_LOGS, "Validation"), exist_ok = True)
writer_training = SummaryWriter(
    log_dir = os.path.join(P_DIR_LOGS, "Training")
)
writer_validation = SummaryWriter(
    log_dir = os.path.join(P_DIR_LOGS, "Validation")
)


# ======== TRAINING ======== #

# initialize training instance
epoch_training = Epoch(
    model,
    utility_model,
    s_phase = "training",
    loss = loss,
    optimizer = optimizer,
    device = S_DEVICE,
    verbose = True,
    writer = writer_training,
)
# initialize validation instance
epoch_validation = Epoch(
    model,
    utility_model,
    s_phase = "validation",
    loss = loss,
    device = S_DEVICE,
    verbose = True,
    writer = writer_validation,
)

#========log neptune hyperparameters======#
neptune_run['learning_rate_max'].log(str(LR_max))
neptune_run['learning_rate_min'].log(str(LR_min))
neptune_run['encoder'].log(S_NAME_ENCODER)
neptune_run['pretrained'].log( "imagenet" )
neptune_run['batch_size_train'].log( str(N_SIZE_BATCH_TRAINING))
neptune_run['batch_size_val'].log( str(N_SIZE_BATCH_VALIDATION))
neptune_run['noise_coefficient'].log( "0.01")

# start training phase
os.makedirs(P_DIR_CKPT, exist_ok = True)
max_score = 0
min_loss=1000

# iterate over epochs
for i in range(1, N_EPOCH_MAX + 1):
    print(f"Epoch: {i} | LR = {round(scheduler.get_last_lr()[0], 8)}")
    d_log_training = epoch_training.run(loader_training, i_epoch = i)
    
    iou_score = round(d_log_training["iou_score"] * 100, 2)
    print(f"IoU = {iou_score}%")
    
    print()
    
    neptune_run["train/iou_score"].append(iou_score)
    neptune_run["train/lr"].append(round(scheduler.get_last_lr()[0], 8))
    neptune_run["train/total_loss"].append(d_log_training["total_loss"])
    neptune_run["train/BCE_only"].append(d_log_training["BCE_only"])
    neptune_run["train/B_only"].append(d_log_training["B"])
    torch.save(model,os.path.join(P_DIR_CKPT, f"model_epoch_{i:0>4}.pth"))

    # log validation performance
    if i % N_STEP_LOG == 0:
        d_log_validation = epoch_validation.run(loader_validation, i_epoch = i)
        
        total_loss = round(d_log_validation["total_loss"] , 4)
        print(f"total = {total_loss}")
        
        iou_score = round(d_log_validation["iou_score"] * 100, 2)
        print(f"IoU = {iou_score}%")

        neptune_run["val/iou_score"].append(iou_score)
        neptune_run["val/total_loss"].append(d_log_validation["total_loss"])
        
        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(model,os.path.join(P_DIR_CKPT, f"best_model_epoch_{i:0>4}.pth"))
            print("Model saved!")

        print()
    scheduler.step()
writer_training.close()
writer_validation.close()
neptune_run.stop()
