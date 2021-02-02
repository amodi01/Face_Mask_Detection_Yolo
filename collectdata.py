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

file_path ="data/facemask/fmdetection.csv"

annotations = sorted(glob('data/facemask/annotations/*.xml'))
print (annotations)
df = []
cnt = 0
for file in annotations:
  prev_filename = file.split('/')[-1].split('.')[0] + '.png'
  filename = str(cnt) + '.png'
  row = []
  parsedXML = ET.parse(file)
  for node in parsedXML.getroot().iter('object'):
    classtype = node.find('name').text
    xmin = int(node.find('bndbox/xmin').text)
    xmax = int(node.find('bndbox/xmax').text)
    ymin = int(node.find('bndbox/ymin').text)
    ymax = int(node.find('bndbox/ymax').text)

    row = [prev_filename, filename, classtype, xmin, xmax, ymin, ymax]
    df.append(row)
  cnt += 1

data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'classtype', 'xmin', 'xmax', 'ymin', 'ymax'])

data[['prev_filename','filename', 'classtype', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv(file_path, index=False)
del data
