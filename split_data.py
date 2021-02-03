#https://medium.com/towards-artificial-intelligence/yolo-v5-object-detection-on-a-custom-dataset-61d478bc08f9
import os, sys, random, shutil
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
import numpy as np
import csv
#import sklearn
from pathlib import Path
from sklearn import preprocessing, model_selection

img_width = 512
img_height = 366

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


df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
print(df_train.shape, df_valid.shape)

src_img_path = "data/facemask/images/"
src_label_path = "data/facemask/annotations/"

train_img_path = "data/images/train/"
train_label_path = "data/labels/train/"

valid_img_path = "data/images/valid/"
valid_label_path = "data/labels/valid/"

#os.mkdir('/dataset/facemask/')
#os.mkdir('/dataset/facemask/images/')
os.makedirs(train_img_path,exist_ok=True)
os.makedirs(valid_img_path,exist_ok=True)

#os.mkdir('/dataset/facemask/labels/')
os.makedirs(train_label_path,exist_ok=True)
os.makedirs(valid_label_path,exist_ok=True)

def split_data(df, img_path, label_path, train_img_path, train_label_path):
  filenames = []
  for filename in df.filename:
    filenames.append(filename)
  filenames = set(filenames)
  
  for filename in filenames:
    yolo_list = []

    for _,row in df[df.filename == filename].iterrows():
      yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

    yolo_list = np.array(yolo_list)
    txt_filename = os.path.join(train_label_path,str(row.prev_filename.split('.')[0])+".txt")
    # Save the .img & .txt files to the corresponding train and validation folders
    np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
    shutil.copyfile(os.path.join(img_path,row.prev_filename), os.path.join(train_img_path,row.prev_filename))
 
## Apply function ## 


split_data(df_train, src_img_path, src_label_path, train_img_path, train_label_path)
split_data(df_valid, src_img_path, src_label_path, valid_img_path, valid_label_path)

print("No. of Training images", len(os.listdir(train_img_path)))
print("No. of Training labels", len(os.listdir(train_label_path)))

print("No. of valid images", len(os.listdir(valid_img_path)))
print("No. of valid labels", len(os.listdir(valid_label_path)))
