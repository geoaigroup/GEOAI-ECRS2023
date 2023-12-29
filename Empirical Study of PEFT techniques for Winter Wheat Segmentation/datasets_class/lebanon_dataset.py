import numpy as np
from torch.utils.data.dataloader import Dataset
import os
import rasterio as rio

import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

#model_params
MEAN = np.array([[[[1165.9398193359375]],[[1375.6534423828125]],[[1429.2191162109375]],[[1764.798828125]],[[2719.273193359375]],[[3063.61181640625]],[[3205.90185546875]],[[3319.109619140625]],[[2422.904296875]],[[1639.370361328125]]]]).astype(np.float32)
STD = np.array([[[[1942.6156005859375]],[[1881.9234619140625]],[[1959.3798828125]],[[1867.2239990234375]],[[1754.5850830078125]],[[1769.4046630859375]],[[1784.860595703125]],[[1767.7100830078125]],[[1458.963623046875]],[[1299.2833251953125]]]]).astype(np.float32)



class TSViT_leb_dataset(Dataset):
    
    def __init__(self,basepath,
                 years=[2016,2017,2018,2019,2020],
                 aois=[0,1,2,3,4],
                 list_of_tile=None,
                 min_pixels_per_image=-1,
                vertical_flip=False,
                horizontal_flip=False,
                upload_image=False):
        self.vertical_flip=vertical_flip
        self.horizontal_flip=horizontal_flip
        self.datalist=[]
        self.basepath=basepath
        self.upload_image=upload_image
        if list_of_tile is None:
          for year in years:
              for aoi in aois:
                  temp_path=os.path.join(basepath,f"year_{year}",f"aoi_0_{aoi}","aoi_masks")
                  temp_path=os.listdir(temp_path)
                  temp_path=list(map(lambda x: (year,aoi,x),temp_path))
                  self.datalist.extend(temp_path)
                  # print(self.datalist)
        else:
          self.datalist=[(a[1],a[2],a[3]) for a in list_of_tile if a[0] >= min_pixels_per_image
                                                                and a[1] in years
                                                                and a[2] in aois]
    def __len__(self):
        return len(self.datalist)
    
    def __str__(self):
        return str(self.datalist)
    
    def set_upload_images(self, value):
        self.upload_image=value
    
    def __getitem__(self, idx):
        months=["11","12","01","02","03","04","05","06","07"]

        year=self.datalist[idx][0]
        aoi=self.datalist[idx][1]

        temp_path=os.path.join(self.basepath,f"year_{self.datalist[idx][0]}",f"aoi_0_{self.datalist[idx][1]}")
        # for tile in os.listdir(os.path.join(temp_path,f"aoi_masks")):
            # print(tile)
        tile=self.datalist[idx][2]
        concatenated_matrix=np.zeros((0,11,24,24))

        tile_data=tile.split(".")[0]
        tile_data=tile_data.split("_")
        aoi_mask_data=os.path.join(temp_path,f"aoi_masks",f"tile_{tile_data[1]}_{tile_data[2]}.tiff")
        mask_data=os.path.join(temp_path,f"masks",f"tile_{tile_data[1]}_{tile_data[2]}.tiff")
        mask_data=rio.open(mask_data).read()
        aoi_mask_data=rio.open(aoi_mask_data).read()
        if self.upload_image:
            img=np.zeros((0,10,24,24))

        for month in months:
            if month not in ["11",'12']:
                path=os.path.join(temp_path,f"month_{month}",tile)
            else :
                path=os.path.join(self.basepath,f"year_{year-1}",f"aoi_0_{aoi}",f"month_{month}",tile)
            image=rio.open(path).read()
            # image=image/MAX.reshape(-1,1,1)
            image=image[[1,2,3,4,5,6,7,8,11,12]]
            if self.upload_image:
                img=np.concatenate([img,image[np.newaxis]],axis=0)
            image=(image-MEAN.reshape(-1,1,1))/STD.reshape(-1,1,1)
            # image=image[[1,2,3,4,5,6,7,8,11,12]]
            xt=np.ones((1,24,24))*(int(month)/12.00001)
            image=np.concatenate([image,xt],axis=0)
            image=image[np.newaxis]
            concatenated_matrix=np.concatenate([concatenated_matrix,image],axis=0)
        
        aoi_mask_data=(aoi_mask_data!=0)[0]
        if year != 2020:
            aoi_mask_data&=mask_data.sum(axis=0)!=0
        mask_data=mask_data[1]!=0
        
        if self.vertical_flip:
            if np.random.uniform()<0.5:
                concatenated_matrix=np.flip(concatenated_matrix,2).copy()
                mask_data=np.flip(mask_data,0).copy()
                aoi_mask_data=np.flip(aoi_mask_data,0).copy()

        if self.horizontal_flip:
            if np.random.uniform()<0.5:
                concatenated_matrix=np.flip(concatenated_matrix,3).copy()
                mask_data=np.flip(mask_data,1).copy()
                aoi_mask_data=np.flip(aoi_mask_data,1).copy()
        if self.upload_image:
            return img,concatenated_matrix, mask_data, aoi_mask_data
        return concatenated_matrix, mask_data, aoi_mask_data





if __name__=="__main__":
    dataset=TSViT_leb_dataset("D:\\GEOAI\\code\\Requested_Tiffs_lcc\\cropped_tiffs_24",
                 years=[2016,2017,2018,2019,2020],
                 aois=[0,1,2,3,4],
                 list_of_tile=None,
                 min_pixels_per_image=-1,
                vertical_flip=False,
                horizontal_flip=False,
                upload_image=False)
    datalist=[]
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        X,y,m=dataset[i]
        datalist.append((m.sum(),*dataset.datalist[i]))
    print(X.shape,y.shape,m.shape)
    # print(y,m)
    print(X.shape,y.sum()!=0,m.sum()!=0)
    print(len(datalist))
    import pickle
    with open("lbdata.pkl","wb") as f:
        pickle.dump(datalist,f)


