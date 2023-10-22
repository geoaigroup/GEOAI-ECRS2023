import os
import cv2
import math

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize

import matplotlib.pyplot as plt

from shapely.geometry import Polygon,box
from tqdm.notebook import tqdm

import warnings

warnings.simplefilter("ignore")

dataset_dir = 'Requested_Tiffs_lcc'


img_dir = os.path.join(dataset_dir,'tiffs')
label_dir = os.path.join(dataset_dir,'WheatDataset_Cleaned')
aoi_path = 'lebanon_region_of_study/wheats_aoi_baalbeck.shp'

main_aoi = gpd.read_file(aoi_path)



def fetch_labels(dir,year):
    return gpd.read_file(f'{dir}/{year}_cleaned/{year}_cleaned.shp')

def fetch_tiff_paths(dir):
    #year_img_dir = f'{dir}/{year}'
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".tiff"):
                paths.append(os.path.join(root,file))
                #print(root,file)
    return sorted(paths)

def map_type(x):
    if x == 'w':
        return 0
    elif x == 'b':
        return 1
    elif x == 'nw':
        return 2
    else:
        return 3

def label_polys(gdf):
    gdf['label'] = gdf['Name'].apply(lambda x:map_type(x))
    return gdf


def reshape_split(image, kernel_size):
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)

    tiled_array = tiled_array.swapaxes(1, 2)

    return tiled_array

def pad_to_fit(img,kernel_size):
    h,w = img.shape[:2]

    h_new = int(math.ceil(h / kernel_size[0]) * kernel_size[0])
    w_new = int(math.ceil(w / kernel_size[1]) * kernel_size[1])
    
    img = cv2.copyMakeBorder(img,top=0,left=0,bottom=h_new-h,right=w_new-w,borderType=cv2.BORDER_CONSTANT,value=0.0)


    return img

def write_tiff(path,data,crs=None,transform=None):
    data = data.transpose(2,0,1)
    with rio.open(
                path,
                mode = 'w+',
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
        dst.write(data)


tile_size = (64,64)
tz = tile_size[0]
save_as_npy = False
mcs = 4 #['w','b','nw','u**']

out_dir = os.path.join(dataset_dir,f'cropped_tiffs_{tz}')

for year in sorted(os.listdir(img_dir)):

    year_gdf = fetch_labels(label_dir,year)
    year_gdf = label_polys(year_gdf)

    year_img_dir = f'{img_dir}/{year}'

    loader = tqdm(os.listdir(year_img_dir))
    for aoi in loader:
        year_aoi_img_dir = os.path.join(year_img_dir,aoi)
        tiff_paths = fetch_tiff_paths(year_aoi_img_dir)
        
        if len(tiff_paths) == 0:
            print(f'Warning {year} has no images for {aoi}!!!!')
            continue
        
        for i,tiff_path in enumerate(tiff_paths):

            month = tiff_path.split('/')[-3]
            loader.set_description(f'Processing {year}-{aoi}-{month} crops...')
            r = rio.open(tiff_path)

            if i==0:
                bounds = r.bounds
                bbox = box(*bounds)

                try:
                    aoi_gdf = year_gdf.to_crs(crs=r.crs).clip(bbox,keep_geom_type=True)#intersection(bbox) 
                except: 
                    aoi_gdf = year_gdf.to_crs(crs=r.crs)
                    aoi_gdf['geometry'] = aoi_gdf['geometry'].buffer(0.0)   
                    aoi_gdf = aoi_gdf.clip(bbox,keep_geom_type=False)
                
                main_aoi_clipped = main_aoi.to_crs(crs=r.crs).clip(bbox,keep_geom_type=True)

            img = r.read().transpose(1,2,0)
            h,w,c = img.shape

            if i ==0:
                transform = rio.transform.from_bounds(*bounds, w,h)

                mask = np.zeros((h,w,mcs),dtype=np.uint8)
                
                for mc in range(mcs):

                    geoms = aoi_gdf.loc[(aoi_gdf['label'] == mc)].intersection(bbox)
                    if len(geoms) == 0:
                        continue
                    mask[:,:,mc] = rasterize(
                        geoms, 
                        all_touched=True,
                        transform=transform, 
                        out_shape=(h, w),
                        default_value=255.0,
                        dtype=np.uint8,
                        fill=0.0)
                
                main_aoi_mask = rasterize(
                        main_aoi_clipped['geometry'], 
                        all_touched=True,
                        transform=transform, 
                        out_shape=(h, w),
                        default_value=255.0,
                        dtype=np.uint8,
                        fill=0.0)
                

                mask = pad_to_fit(mask,tile_size)
                mask = reshape_split(mask,tile_size)

                main_aoi_mask = pad_to_fit(main_aoi_mask,tile_size)
                main_aoi_mask = reshape_split(main_aoi_mask,tile_size)

            img = pad_to_fit(img,tile_size)
            img = reshape_split(img,tile_size)

            rows,cols = img.shape[:2]
            #fig,axs = plt.subplots(rows*3,cols,figsize=(5*cols,5*rows*3))

            for row in range(rows):
                for col in range(cols):
                    tile_id = f'tile_{row}_{col}'
                    aoi_mask_tile = main_aoi_mask[row,col,...]
                    if aoi_mask_tile.max() <=0:
                        continue

                    if i == 0 and year!="year_2015": 
                        out_now = f'{out_dir}/{year}/{aoi}/masks'
                        out_now2 = f'{out_dir}/{year}/{aoi}/aoi_masks'

                        mask_tile = mask[row,col,...]
                    
                        os.makedirs(out_now,exist_ok=True)
                        os.makedirs(out_now2,exist_ok=True)
                        #axs[row+rows,col].imshow(mask_tile[:,:,:3])
                        if save_as_npy:
                            np.save(os.path.join(out_now,tile_id+'.npy'),mask_tile)
                            np.save(os.path.join(out_now2,tile_id+'.npy'),aoi_mask_tile)
                        else:
                            write_tiff(os.path.join(out_now,tile_id+'.tiff'),mask_tile,crs=None,transform=None)
                            write_tiff(os.path.join(out_now2,tile_id+'.tiff'),aoi_mask_tile,crs=None,transform=None)

                    img_tile = img[row,col,...]

                    #rgb = img_tile[:,:,[3,2,1]]
                    #rgb = rgb*2.5
                    #rgb_norm = (rgb) / 10000
                    #axs[row,col].imshow(rgb_norm)
                    
                    out_now = f'{out_dir}/{year}/{aoi}/{month}'
                    
                    os.makedirs(out_now,exist_ok=True)

                    if save_as_npy:
                        np.save(os.path.join(out_now,tile_id+'.npy'),img_tile)
                    else:
                        write_tiff(os.path.join(out_now,tile_id+'.tiff'),img_tile,crs=None,transform=None)
