#https://medium.com/towards-artificial-intelligence/yolo-v5-object-detection-on-a-custom-dataset-61d478bc08f9
import os, sys, random, shutil
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing, model_selection

img_width = 640
img_height = 480

def width(df):
  return int(df.xmax - df.xmin)
def height(df):
  return int(df.ymax - df.ymin)
def x_center(df):
  return int(df.xmin + (df.width/2))
def y_center(df):
  return int(df.ymin + (df.height/2))
def w_norm(df):
  return df/img_width
def h_norm(df):
  return df/img_height

file_path ="data/facemask/fmdetection.csv"

df = pd.read_csv(file_path)

le = preprocessing.LabelEncoder()
le.fit(df['classtype'])
print(le.classes_)
labels = le.transform(df['classtype'])
df['labels'] = labels

df['width'] = df.apply(width, axis=1)
df['height'] = df.apply(height, axis=1)

df['x_center'] = df.apply(x_center, axis=1)
df['y_center'] = df.apply(y_center, axis=1)

df['x_center_norm'] = df['x_center'].apply(w_norm)
df['width_norm'] = df['width'].apply(w_norm)

df['y_center_norm'] = df['y_center'].apply(h_norm)
df['height_norm'] = df['height'].apply(h_norm)

