import rasterio as rio
import os
import numpy as np
from datetime import datetime
import tqdm

from config import MUNICH_DATA_PATH,PROCCESSED_DATA_PATH

"""This code process the munich dataset from 480mx480m to the size of tx10x24x24 as given in the PASTIS dataset
where 
t: temporal data
10: number of S2 bands used 
24x24: 240x240m image as trained in the TSViT work
original data is as follows: 
munich480:
    \\data16
        \\tile_id
            \\date_resolutions
    \\data17
        \\tile_id
            \\date_resolutions
"""


datapath=MUNICH_DATA_PATH
preprocessed_data_path=PROCCESSED_DATA_PATH

years=[16,17]
data=[]#will be filled as follow: (year,tile_id,{set of all dates for this tile})
for year in years:
    for tile in os.listdir(f"{datapath}/data{year}"):
        if not os.path.isdir(f"{datapath}/data{year}/{tile}"):
            continue
        datum=[a.split("_")[0] for a in os.listdir(f"{datapath}/data{year}/{tile}") if not a.startswith("y")] 
        data.append((year,tile,set(datum)))
   

def getimage(year,tile,data):
    """takes the 10m and 20m resolution data and reorder them them to fit witht the TSViT images"""
    reorder_list=[2,1,0,4,5,6,3,7,8,9]

    path=f"{datapath}/data{year}/{tile}/{data}"
    path10=path+"_10m.tif"
    path20=path+"_20m.tif"
    
    with rio.open(path10,'r') as f:
        x10=f.read(out_shape=(48,48))
    with rio.open(path20,'r') as f:
        x20=f.read(out_shape=(48,48))
    
    x=np.concatenate([x10,x20],axis=0)
    x=x[reorder_list]
    return x
    
def day_of_year(data):
    """takes yyyymmdd and transform it into day_of_year """
    year=int(data[:4])
    month=int(data[4:6])
    day=int(data[6:])
    date=datetime(year,month,day)
    return date.timetuple().tm_yday

def get_y(year,tile):
    """gets the y for each tile"""
    path=f"{datapath}/data{year}/{tile}/y.tif"
    with rio.open(path,'r') as f:
        return f.read()
    

#now we start collecting the data to transofrm into 24x24 tiles
os.makedirs(preprocessed_data_path)
for year,tile,datum in tqdm.tqdm(data):
    respath=os.path.join(preprocessed_data_path,f"{year}_{tile}")#the result of prerocessing will be saved here
    try:
        X=np.zeros((0,10,48,48))
        doys=[]
        #looping over to get all images for 1 tile in the form of 48x48
        for d in datum:
            img=getimage(year=year,tile=tile,data=d).reshape((1,10,48,48))
            doy=day_of_year(d)
            X=np.concatenate([img,X],axis=0)
            doys.append(doy)
        y=get_y(year,tile)

        #split into 4 24x24 images
        np.savez(respath+"_1.tif.npz",X=X[:,:,:24,:24],y=y[:,:24,:24],doy=doys)
        np.savez(respath+"_2.tif.npz",X=X[:,:,:24,24:],y=y[:,:24,24:],doy=doys)
        np.savez(respath+"_3.tif.npz",X=X[:,:,24:,:24],y=y[:,24:,:24],doy=doys)
        np.savez(respath+"_4.tif.npz",X=X[:,:,24:,24:],y=y[:,24:,24:],doy=doys)
    except rio.RasterioIOError:
        print(tile,year,"Error")