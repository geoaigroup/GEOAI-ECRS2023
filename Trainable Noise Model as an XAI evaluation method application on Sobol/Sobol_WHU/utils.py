
import torch
from torch.nn.functional import interpolate

import os
import cv2
import random

import numpy as np

from math import ceil
from skimage.segmentation import watershed
from skimage.measure import label


def make_dir(path):
    os.makedirs(path,exist_ok=True)

#load model util
def load_model(model,path):
    checkpoint = torch.load(f'{path}/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

#simple send to tensor
def totensor(x):
    return torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float()

def normalize(x):
    x = x.astype(np.float64)
    x -= (0.5 * 255.0)
    x /= (0.5 * 255.0)
    return x
    
#padding - unpadding utils
def get_pads(L,div=32):
    if L % div == 0:
        x1,x2 = 0,0
    else:
        L_pad = ceil(L / div) * div
        dL = L_pad - L
        x1 = max(1,dL//2)
        x2 = dL - x1
    return x1,x2

def ratio_resize_pad(img,ratio=None,div=32):

    h_rsz,w_rsz = h_orig,w_orig = img.shape[:2]

    if ratio is not None:
        if ratio != 1.0:
            h_rsz = ceil(h_orig * ratio)
            w_rsz = ceil(w_orig * ratio)
            img = cv2.resize(img,(w_rsz,h_rsz))

    t,b = get_pads(h_rsz,div)
    l,r = get_pads(w_rsz,div)
    img = cv2.copyMakeBorder(img,t,b,l,r,borderType=cv2.BORDER_CONSTANT,value=0.0)

    info = {'orig_size' : (h_orig,w_orig),'pads':(t,b,l,r)}
    return img,info
    
def unpad_resize(img,info):
    h,w = img.shape[2:]

    t,b,l,r = info['pads']
    orig_size = info['orig_size']

    img = img[:,:,t:h-b,l:w-r]
    if h != orig_size[0] or w != orig_size[1]:
        #img = cv2.resize(img,orig_size).astype(np.uint8)
        img = interpolate(img,size=orig_size)
    return img


#multi class post processing
def noise_filter(washed,mina):
    values = np.unique(washed)
    for val in values[1:]:
        area = (washed[washed == val]>0).sum()
        if(area<=mina):  
            washed[washed == val] = 0
    return washed
    
def post_process(pred,thresh = 0.5,thresh_b = 0.6,mina=100,mina_b=50):
    if len(pred.shape) < 2:
        return None
    if len(pred.shape) == 2:
        pred = pred[...,np.newaxis]
    
    ch = pred.shape[2]
    buildings = pred[...,0]
    if ch > 1:
        borders = pred[...,1]
        nuclei = buildings * (1.0 - borders)

        if ch == 3:
            spacing = pred[...,2]
            nuclei *= (1.0 - spacing)

        basins = label(nuclei>thresh_b,background = 0, connectivity = 2)
        if mina_b > 0:
            basins = noise_filter(basins, mina = mina_b)
            basins = label(basins,background = 0, connectivity = 2)

        washed = watershed(image = -buildings,
                           markers = basins,
                           mask = buildings>thresh,
                           watershed_line=False)

    elif(ch == 1):
        washed  = buildings > thresh 


    washed = label(washed,background = 0, connectivity = 2)
    washed = noise_filter(washed, mina=mina)
    washed = label(washed,background = 0, connectivity = 2)
        
    return washed

#visualization utils
def mask2rgb(mask,max_value=1.0):
    shape = mask.shape
    if len(shape) == 2:
        mask = mask[:,:,np.newaxis]
    h,w,c = mask.shape
    if c == 3:
        return mask
    if c == 4:
        return mask[:,:,:3]
    
    if c > 4:
        raise ValueError
    
    padded = np.zeros((h,w,3),dtype=mask.dtype)
    padded[:,:,:c] = mask
    padded = (padded * max_value).astype(np.uint8)
    
    return padded


def make_rgb_mask(mask,color=(255,0,0)):
    h,w = mask.shape[:2]
    rgb = np.zeros((h,w,3),dtype=np.uint8)
    rgb[mask == 1.0,:] = color
    return rgb

def overlay_rgb_mask(img,mask,sel,alpha):

    sel = sel == 1.0
    img[sel,:] = img[sel,:] * (1.0 - alpha) + mask[sel,:] * alpha
    return img

def overlay_instances_mask(img,instances,cmap,alpha=0.9):
    h,w = img.shape[:2]
    overlay = np.zeros((h,w,3),dtype=np.float32)

    _max = instances.max()
    _cmax = cmap.shape[0]
    

    if _max == 0:
        return img
    elif _max > _cmax:
        indexes = [(i % _cmax) for i in range(_max)]    
    else:
        indexes = random.sample(range(0,_cmax),_max)
    
    for i,idx in enumerate(indexes):
        overlay[instances == i+1,:] = cmap[idx,:]
    
    overlay = (overlay * 255.0).astype(np.uint8)
    viz = overlay_rgb_mask(img,overlay,instances>0,alpha=alpha)
    return viz
