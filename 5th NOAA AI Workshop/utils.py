import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import geopandas as gpd
import os
import json
import glob
from tqdm import tqdm
import shapely.geometry as sg
from shapely import affinity
from shapely.geometry import Point, Polygon
import random
from PIL import Image, ImageDraw
from skimage import measure
import rasterio
from rasterio.features import geometry_mask
#from metrics import DiceScore,IoUScore
import pandas as pd
import gc
import shutil
import fiona
width=512
height=512

def show_mask(mask,ax,random_color=False,s=""):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            if s=="gt":
                color = np.array([30/255, 144/255, 255/255, 0.5])
            elif s=="whu":
               color = np.array([0/255, 255/255, 0/255, 0.4])
            elif s=="pred":
                color = np.array([255/255, 0/255, 0/255, 0.5])
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
            #return mask_image

def show_mask_box(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
       color = np.array([30/255, 144/255, 255/255, 0.6])
       h, w = mask.shape[-2:]
       mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
       ax.imshow(mask_image)
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))



def create_boxes(geo):
    tile_boxes=[]
    for p in geo:
          inbox=[]
          poly=p
          xmin,ymin,xmax,ymax=poly.bounds
          inbox=[xmin,ymin,xmax,ymax]

          tile_boxes.append(inbox)
    return tile_boxes



def binary_mask_to_polygon(binary_mask):
    # Find contours in the binary mask
    contours = measure.find_contours(binary_mask, 0.5)
    # Get the largest contour (in case there are multiple objects)
    max_contour = max(contours, key=len)

    # Convert the contour points to a polygon (list of (x, y) coordinates)
    polygon = Polygon([(int(point[1]), int(point[0])) for point in max_contour])

    return polygon

def convert_polygon_to_mask_batch(geo):
  gtmask=[]
  for orig_row in geo:
    polygon=[]
    if orig_row.geom_type=="Polygon":
        for point in orig_row.exterior.coords:
          polygon.append(point)
        img = Image.new('L', (width, height),0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        img=np.array(img)
        gtmask.append(img)
    else:
        for x in orig_row.geoms:
          for point in x.exterior.coords:
            polygon.append(point)
        img = Image.new('L', (width, height),0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        img=np.array(img)
        gtmask.append(img)
  return gtmask


def generate_in_negative_points(geo):
            all_points=[]
            all_labels=[]

            step=5
            n=5
            for polygon in geo:
                input_point=[]
                input_label=[]
                min_x, min_y, max_x, max_y = polygon.bounds
                while len(input_point) < n:
                  # Generate a random point within the bounding box
                  random_point = Point(random.randrange(min_x, max_x,step), random.randrange(min_y, max_y,step))

                  if not polygon.contains(random_point):
                      x=random_point.x
                      y=random_point.y
                      k=[x,y]
                      input_point.append(k)
                      input_label.append(0)

                all_points.append(input_point)
                all_labels.append(input_label)

            all_points=torch.from_numpy(np.array(all_points)).cuda()
            all_labels=torch.from_numpy(np.array(all_labels)).cuda()
            return all_points,all_labels

def _getArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])
def generate_out_negative_points(geo):
            input_point=[]
            input_label=[]
            bounding_boxes=create_boxes(geo)
            n=len(geo)

            for box in bounding_boxes:
                area=_getArea(box)
                print("box",box)
                if area >= 262000:
                    a1=0
                    a2=0
                    return a1,a2


            while len(input_point) < n:
              random_points = np.random.randint(0, 512, size=(n, 2))
              for point in random_points:
                is_inside_bbox = any(bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3] for bbox in bounding_boxes)
                if not is_inside_bbox and len(input_point) < n:
                    x=point[0]
                    y=point[1]
                    k=[x,y]
                    input_point.append(k)
                    input_label.append(0)


            input_point=torch.from_numpy(np.array(input_point)).unsqueeze(1).cuda()
            input_label=torch.from_numpy(np.array(input_label)).unsqueeze(1).cuda()
            return input_point,input_label

def generate_negative_points(geo):
            input_point=[]
            input_label=[]
            step=70
            n=len(geo)
            # if n>=60:
            #     n=20
            while len(input_point) < n:
              # Generate a random point within the bounding box
              random_point = Point(random.randrange(0, 512,step), random.randrange(0, 512,step))

              # Check if the random point is outside all polygons
              if all(not polygon.contains(random_point) for polygon in geo):
                  x=random_point.x
                  y=random_point.y
                  k=[x,y]
                  input_point.append(k)
                  input_label.append(0)
            return input_point,input_label


def generate_random_points_polygon(geo,flag=""):
            all_points=[]
            all_labels=[]
            for p in geo:
                n=5
                step=1
                area=p.area/512
                if area<2:
                    step=3

                elif area>=2 and area<5:
                    step=7

                elif area>=5 and area<11:
                    step=9

                elif area>=11:
                    step=10
                input_point=[]
                input_label=[]
                min_x, min_y, max_x, max_y = p.bounds
                while len(input_point) < n:
                    point = Point(random.randrange(min_x, max_x,step), random.randrange(min_y, max_y,step))
                    if p.contains(point):
                        x=point.x
                        y=point.y
                        k=[x,y]
                        input_point.append(k)
                        input_label.append(1)
                        #print(points)
                if flag=="rep":
                    rep_point=p.representative_point()
                    ix=rep_point.x
                    iy=rep_point.y
                    i=[ix,iy]
                    input_point.append(i)
                    input_label.append(1)
                all_points.append(input_point)
                all_labels.append(input_label)

            if flag=="negative":
                in_p,in_l=generate_negative_points(geo)
                for index,(_,_) in enumerate(zip(all_points,all_labels)):
                    all_points[index].append(in_p[index])
                    all_labels[index].append(in_l[index])

            input_point1=torch.from_numpy(np.array(all_points)).cuda()
            input_label1=torch.from_numpy(np.array(all_labels)).cuda()

            return input_point1,input_label1



def create_list_points(geo,name,flag=""):
                all_points=[]
                all_labels=[]
                for p in geo:
                  input_point=[]
                  input_label=[]
                  if p.geom_type=="Polygon":
                    point=p.representative_point()
                    x=point.x
                    y=point.y
                    i=[x,y]
                    input_point.append(i)
                    input_label.append(1)
                  else:
                   for pp in p.geoms:
                    point=pp.representative_point()
                    x=point.x
                    y=point.y
                    i=[x,y]
                    input_point.append(i)
                    input_label.append(1)
                    break
                  all_points.append(input_point)
                  all_labels.append(input_label)
                if flag=="negative":
                    in_p,in_l=generate_negative_points(geo)
                    for index,(_,_) in enumerate(zip(all_points,all_labels)):
                        all_points[index].append(in_p[index])
                        all_labels[index].append(in_l[index])



                all_points=torch.from_numpy(np.array(all_points)).cuda()
                all_labels=torch.from_numpy(np.array(all_labels)).cuda()
                return all_points,all_labels