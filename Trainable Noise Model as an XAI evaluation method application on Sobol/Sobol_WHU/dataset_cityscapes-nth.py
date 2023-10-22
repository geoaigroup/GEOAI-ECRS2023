import torchvision
import numpy as np
import cv2
import lookup_table as lut


class DatasetCityscapesSemantic(torchvision.datasets.Cityscapes):
    def __init__(self, device, excluded ,*args, **kwargs):

        super().__init__(*args, **kwargs, target_type = "semantic")
        self.excluded = excluded
        self.device = device
        
        #exclude images and target where gtmask for road is missing
        
        new_images , new_targets=[],[]
        for i,file_path in enumerate(self.images):
            if file_path in self.excluded:
                continue
            new_images.append(file_path)
            new_targets.append(self.targets[i])
       
        self.images = new_images
        self.targets = new_targets
        print(len(self.images), len(self.targets))

        
        
        
        
  
        # setup lookup tables for class/color conversions
        l_key_id, l_key_trainid, l_key_color = self._get_class_properties()
        ar_u_key_id = np.asarray(l_key_id, dtype = np.uint8)
        ar_u_key_trainid = np.asarray(l_key_trainid, dtype = np.uint8)
        ar_u_key_color = np.asarray(l_key_color, dtype = np.uint8)
        _, self.th_i_lut_id2trainid = lut.get_lookup_table(
            ar_u_key = ar_u_key_id,
            ar_u_val = ar_u_key_trainid,
            v_val_default = 19,  # default class is 19 - background
            device = self.device,
        )
        _, self.th_i_lut_trainid2id = lut.get_lookup_table(
            ar_u_key = ar_u_key_trainid,
            ar_u_val = ar_u_key_id,
            v_val_default = 0,  # default class is 0 - unlabeled
            device = self.device,
        )
        _, self.th_i_lut_trainid2color = lut.get_lookup_table(
            ar_u_key = ar_u_key_trainid,
            ar_u_val = ar_u_key_color,
            v_val_default = 0,  # default color is black
            device = self.device,
        )
    
    def _get_class_properties(self):
        # iterate over named tuples (nt)
        l_key_id = list()
        l_key_trainid = list()
        l_key_color = list()
        # append classes
        for nt_class in self.classes:
            if nt_class.train_id in [-1, 255]:
                continue
            l_key_id.append([nt_class.id])
            l_key_trainid.append([nt_class.train_id])
            l_key_color.append(nt_class.color)
        # append class background
        l_key_id.append([0])
        l_key_trainid.append([19])
        l_key_color.append([0, 0, 0])
        return l_key_id, l_key_trainid, l_key_color
    
    def __getitem__(self, index):
        # read images
        p_image = self.images[index]
        p_target = self.targets[index][0]  # 0 is index of semantic target type
        image = cv2.imread(p_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = cv2.imread(p_target, cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            transformed = self.transform(image = image, mask = target)
            image = transformed["image"]
            target = transformed["mask"]
        return image, target, p_image, p_target
