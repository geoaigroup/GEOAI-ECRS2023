import torchvision
import numpy as np
import cv2
import torch.utils.data as data
import glob
import os
import torch


# class DatasetCityscapesSemantic(torchvision.datasets.Cityscapes):
#     def __init__(self, device, excluded ,*args, **kwargs):

#         super().__init__(*args, **kwargs, target_type = "semantic")
#         self.excluded = excluded
#         self.device = device
        
#         #exclude images and target where gtmask for road is missing
        
#         new_images , new_targets=[],[]
#         for i,file_path in enumerate(self.images):
#             if file_path in self.excluded:
#                 continue
#             new_images.append(file_path)
#             new_targets.append(self.targets[i])
       
#         self.images = new_images
#         self.targets = new_targets
#         print(len(self.images), len(self.targets))


    
#     def __getitem__(self, index):
#         # read images
#         p_image = self.images[index]
#         p_target = self.targets[index][0]  # 0 is index of semantic target type
#         image = cv2.imread(p_image)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         target = cv2.imread(p_target, cv2.IMREAD_UNCHANGED)
#         if self.transform is not None:
#             transformed = self.transform(image = image, mask = target)
#             image = transformed["image"]
#             target = transformed["mask"]
#         return image, target, p_image, p_target



class DataLoaderSegmentation(data.Dataset):
    def __init__(self, image_path, mask_path, transform, device):
        super(DataLoaderSegmentation, self).__init__()
        self.transform = transform
        self.device = device
        self.img_files = glob.glob(os.path.join(image_path,'*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(mask_path,os.path.basename(img_path)) )
        print("length of images: ", len(self.img_files))
        print("length of masks: ", len(self.mask_files))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]  
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            target = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if self.transform is not None:
                transformed = self.transform(image = image, mask = target)
                image = transformed["image"]
                target = transformed["mask"]
            return torch.from_numpy(image).float(), torch.from_numpy(target).float(), img_path, mask_path

    def __len__(self):
        return len(self.img_files)


