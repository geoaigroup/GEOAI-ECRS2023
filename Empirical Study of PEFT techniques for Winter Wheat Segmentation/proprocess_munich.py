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

"""


datapath=MUNICH_DATA_PATH
preprocessed_data_path=PROCCESSED_DATA_PATH

years=[16,17]
data=[]
for year in years:
    for tile in os.listdir(f"{datapath}/data{year}"):
        if not os.path.isdir(f"{datapath}/data{year}/{tile}"):
            continue
        datum=[a.split("_")[0] for a in os.listdir(f"{datapath}/data{year}/{tile}") if not a.startswith("y")]
        data.append((year,tile,set(datum)))

B10,B20,B60=[3,2,1,7],[4,5,6,9],[0,11,12]
reorder_list=[2,1,0,4,5,6,3,7,8,9]
def getimage(year,tile,data):
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
    year=int(data[:4])
    month=int(data[4:6])
    day=int(data[6:])
    date=datetime(year,month,day)
    return date.timetuple().tm_yday
def get_y(year,tile):
    path=f"{datapath}/data{year}\\{tile}\\y.tif"
    with rio.open(path,'r') as f:
        return f.read()
    

respath=os.path.join(preprocessed_data_path,f"{16}_{10000}")

os.makedirs(preprocessed_data_path,exist_ok=True)
for year,tile,datum in tqdm.tqdm(data):
    respath=os.path.join(preprocessed_data_path,f"{year}_{tile}")
    try:
        X=np.zeros((0,10,48,48))
        doys=[]
        for d in datum:
            img=getimage(year=year,tile=tile,data=d).reshape((1,10,48,48))
            doy=day_of_year(d)
            X=np.concatenate([img,X],axis=0)
            doys.append(doy)
        y=get_y(year,tile)

        
        np.savez(respath+"_1.tif.npz",X=X[:,:,:24,:24],y=y[:,:24,:24],doy=doys)
        np.savez(respath+"_2.tif.npz",X=X[:,:,:24,24:],y=y[:,:24,24:],doy=doys)
        np.savez(respath+"_3.tif.npz",X=X[:,:,24:,:24],y=y[:,24:,:24],doy=doys)
        np.savez(respath+"_4.tif.npz",X=X[:,:,24:,24:],y=y[:,24:,24:],doy=doys)
    except rio.RasterioIOError:
        print(tile,year,"Error")