import numpy as np
from torch.utils.data.dataloader import Dataset
import os

from datatransform import PASTIS_segmentation_transform
from config import TSVIT_config

import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

#model_params
MEAN = np.array([[[[1165.9398193359375]],[[1375.6534423828125]],[[1429.2191162109375]],[[1764.798828125]],[[2719.273193359375]],[[3063.61181640625]],[[3205.90185546875]],[[3319.109619140625]],[[2422.904296875]],[[1639.370361328125]]]]).astype(np.float32)
STD = np.array([[[[1942.6156005859375]],[[1881.9234619140625]],[[1959.3798828125]],[[1867.2239990234375]],[[1754.5850830078125]],[[1769.4046630859375]],[[1784.860595703125]],[[1767.7100830078125]],[[1458.963623046875]],[[1299.2833251953125]]]]).astype(np.float32)



class munich_dataset(Dataset):
    
    def __init__(self,basepath,
                 tiles,
                vertical_flip=False,
                horizontal_flip=False,
                padding_length=None):
        
        self.vertical_flip=vertical_flip
        self.horizontal_flip=horizontal_flip
        self.datalist=tiles
        self.basepath=basepath
        config=TSVIT_config.copy()
        if padding_length!=None:
            config["max_seq_len"]=padding_length
        self.transform=PASTIS_segmentation_transform(config,False)

        self.first_v_flip=False
        self.first_h_flip=False
        self.first_no_v_flip=False
        self.first_no_h_flip=False
    def __len__(self):
        return len(self.datalist)
    
    def set_upload_images(self, value):
        self.upload_image=value
    
    def __getitem__(self, idx):

        path=os.path.join(self.basepath,self.datalist[idx])
        tile=np.load(path)
        y=tile["y"]
        # print(tile["X"].shape)
        tile={
            "img":tile["X"].astype(np.float64),
            "labels":np.ones_like(y).astype(np.float64),
            "doy":tile["doy"].astype(np.float64),
            }
        
        tile=self.transform(tile)

        X=tile["inputs"]
        return X,y.astype(np.float64).reshape((24,24))
    
    def __str__(self):
        return self.datalist
    
if __name__=="__main__":
    from tqdm import tqdm
    from torch.utils.data.dataloader import DataLoader
    base_path="D:\\GEOAI\\datasets\\munich_24"
    cnt=0
    dataset=munich_dataset(base_path,os.listdir(base_path))
    X,y=dataset[3]


    for X,y in tqdm(DataLoader(dataset)):
        cnt+=1
        print(X.shape,y.shape)

        if cnt==100:
            break


        