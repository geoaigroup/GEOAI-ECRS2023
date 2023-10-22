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

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# ======== CONFIGURATION ======== #

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
S_EXPERIMENT = "UNET_EfficientNetB4_CE"
P_DIR_DATA = "/home/jamada/jupyterlab/eo-xai/CityScapes_Sementic_Segmentation/datasets/cityscapes/"
P_DIR_CKPT = os.path.join("/home/jamada/jupyterlab/eo-xai/CityScapes_Sementic_Segmentation/", S_EXPERIMENT, "Checkpoints")
P_DIR_LOGS = os.path.join("/home/jamada/jupyterlab/eo-xai/CityScapes_Sementic_Segmentation/", S_EXPERIMENT, "Logs")
P_DIR_EXPORT = os.path.join("/home/jamada/jupyterlab/eo-xai/CityScapes_Sementic_Segmentation/", S_EXPERIMENT, "Export")
S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S_NAME_ENCODER = "efficientnet-b4"
S_NAME_WEIGHTS = "imagenet"
N_EPOCH_MAX = 50
N_SIZE_BATCH_TRAINING = 4  # training batch size
N_SIZE_BATCH_VALIDATION = 4  # validation batch size
N_SIZE_BATCH_TEST = 4  # test batch size
N_SIZE_PATCH = 512 # patch size for random crop
N_STEP_LOG = 5  # evaluate on validation set and save model every N iterations
N_WORKERS = 8  # to be adapted for each system




# U-Net model
model = smp.Unet(
   encoder_name = S_NAME_ENCODER,
   encoder_weights = S_NAME_WEIGHTS,
   in_channels = 3,
   classes = 20,
)

# setup input normalization
preprocess_input = get_preprocessing_fn(
    encoder_name = S_NAME_ENCODER,
    pretrained = S_NAME_WEIGHTS,
)
# setup input augmentations (for cropped and full images)
transform_crop = A.Compose([
    A.RandomCrop(N_SIZE_PATCH, N_SIZE_PATCH),
#    A.HorizontalFlip(p=0.25),
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])
transform_full = A.Compose([
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])

dataset_test = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "test",
    mode = "fine",
    transform = transform_full,
    device = S_DEVICE,
)

loader_test = DataLoader(
    dataset_test,
    batch_size = N_SIZE_BATCH_TEST,
    shuffle = False,
    num_workers = N_WORKERS,
)


# setup loss
loss = torch.nn.CrossEntropyLoss()
loss.__name__ = "ce_loss"


# ======== TEST ======== #

print("\n==== TEST PHASE====\n")
# create export directory
os.makedirs(P_DIR_EXPORT, exist_ok = True)
# load best model
p_model_best = sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[-1]
print(f"Loading following model: {p_model_best}")
model = torch.load(p_model_best)
# initialize test instance
test_epoch = Epoch(
    model,
    s_phase = "test",
    loss = loss,
    p_dir_export = P_DIR_EXPORT,
    device = S_DEVICE,
    verbose = True,
)
test_epoch.run(loader_test)

# remove intermediate checkpoints
for model_checkpoint in sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[:-1]:
    os.remove(model_checkpoint)
