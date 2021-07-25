# creating all the directories we need

!mkdir Data
!mkdir actual_data
!mkdir actual_data/train
!mkdir actual_data/test
!mkdir actual_data/train/0
!mkdir actual_data/train/1

# installing necessities

!pip install pyunpack
!pip install patool

# importing all the packages needed

import os
import tifffile
import numpy as np
import pandas as pd
from pyunpack import Archive
import matplotlib.pyplot as plt
import glob
import random
import shutil
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# Ectracting all the GEO files GEO1,GEO2,GRO5,GEO6 into the created Data directory

Archive('/content/drive/MyDrive/geo/saved_geo1.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo2.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo3.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo4.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo5.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo6.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo7.zip').extractall('/content/Data')
Archive('/content/drive/MyDrive/geo/saved_geo8.zip').extractall('/content/Data')

# appending file names in lists 

geo1_0, geo1_1 = [], []
for files in glob.glob("/content/Data/saved_geo1/*"):
  if files[-5]=='0':
    geo1_0.append(files)
  else:
    geo1_1.append(files)
    
geo2_0, geo2_1 = [], []
for files in glob.glob("/content/Data/saved_geo2/*"):
  if files[-5]=='0':
    geo2_0.append(files)
  else:
    geo2_1.append(files)
    
geo3_0, geo3_1 = [], []
for files in glob.glob("/content/Data/saved_geo3/*"):
  if files[-5]=='0':
    geo3_0.append(files)
  else:
    geo3_1.append(files)
    
geo4_0, geo4_1 = [], []
for files in glob.glob("/content/Data/saved_geo4/*"):
  if files[-5]=='0':
    geo4_0.append(files)
  else:
    geo4_1.append(files)
    
geo5_0, geo5_1 = [], []
for files in glob.glob("/content/Data/saved_geo5/*"):
  if files[-5]=='0':
    geo5_0.append(files)
  else:
    geo5_1.append(files)
    
geo6_0, geo6_1 = [], []
for files in glob.glob("/content/Data/saved_geo6/*"):
  if files[-5]=='0':
    geo6_0.append(files)
  else:
    geo6_1.append(files)
    
geo7_0, geo7_1 = [], []
for files in glob.glob("/content/Data/saved_geo7/*"):
  if files[-5]=='0':
    geo7_0.append(files)
  else:
    geo7_1.append(files)
    
geo8_0, geo8_1 = [], []
for files in glob.glob("/content/Data/saved_geo8/*"):
  if files[-5]=='0':
    geo8_0.append(files)
  else:
    geo8_1.append(files)
    
# checking how many files are present in the folder for each classes.

label_1 = geo1_1 + geo2_1 + geo3_1 + geo4_1 + geo5_1 + geo6_1 + geo7_1 + geo8_1
label_0 = geo1_0 + geo2_0 + geo3_0 + geo4_0 + geo5_0 + geo6_0 + geo7_0 + geo8_0

# printing the stats

print('Total number of images with label 1 : ',len(label_1))
print('Total number of images with label 0 : ',len(label_0))

# using ramdom library to shuffle the filenames from the directories for random sampling

random.shuffle(geo1_0)
random.shuffle(geo2_0)
random.shuffle(geo3_0)
random.shuffle(geo4_0)
random.shuffle(geo5_0)
random.shuffle(geo6_0)
random.shuffle(geo7_0)
random.shuffle(geo8_0)

random.shuffle(geo1_1)
random.shuffle(geo2_1)
random.shuffle(geo3_1)
random.shuffle(geo4_1)
random.shuffle(geo5_1)
random.shuffle(geo6_1)
random.shuffle(geo7_1)
random.shuffle(geo8_1)

# we are splitting the data in such a way that for each class the images in the final data are equal taking all of the class 1 data and selecting the class 0 data
# for that we need to take into consideration how many files each geo image has for  class 1 and proportionally we select class 0 images from the particular GEO image.

# for example if i have 100 images total in the GEO1 for class 1and i am diving the data 70%-30% i have to take 70 images for train and 30 for test i will do the same with the 0 class selecting random 70 images as train and 30 test.
# now if i have 200 images for class 1 in GEO2 i will take 140 images from class 0 and class 1 in the train dat and 60 images from class 0 and class 1 in the test data.

# splitting into train-test as 70%-30% so we need to take first 70% part of the randomly shuffles files in the train for each class .
# similarly we need to take next 30% of the randomly shuffles files for the test data for each class.

train_ratio = 0.7

# calculating how many files we have to take from each directory to consititute the train data for the class 0.

m1 = int(len(geo1_1)*train_ratio)
m2 = int(len(geo2_1)*train_ratio) 
m3 = int(len(geo3_1)*train_ratio) 
m4 = int(len(geo4_1)*train_ratio) 
m5 = int(len(geo5_1)*train_ratio) 
m6 = int(len(geo6_1)*train_ratio) 
m7 = int(len(geo7_1)*train_ratio) 
m8 = int(len(geo8_1)*train_ratio) 

# taking the first 70% from each GEO image for class 0.

files1 = geo1_0[:m1]
files2 = geo2_0[:m2]
files3 = geo3_0[:m3]
files4 = geo4_0[:m4]
files5 = geo5_0[:m5]
files6 = geo6_0[:m6]
files7 = geo7_0[:m7]
files8 = geo8_0[:m8]

# moving the files selected into the train/0 directory.
 
for f in tqdm(files1):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files2):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files3):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files4):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files5):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files6):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files7):
  shutil.move(f,'/content/actual_data/train/0')
for f in tqdm(files8):
  shutil.move(f,'/content/actual_data/train/0')

print('*'*100)

# calculating how many files we have to take from each directory to consititute the test data for the class 0.

files1 = geo1_0[m1:len(geo1_1)-1]
files2 = geo2_0[m2:len(geo2_1)-1]
files3 = geo3_0[m3:len(geo3_1)-1]
files4 = geo4_0[m4:len(geo4_1)-1]
files5 = geo5_0[m5:len(geo5_1)-1]
files6 = geo6_0[m6:len(geo6_1)-1]
files7 = geo7_0[m7:len(geo7_1)-1]
files8 = geo8_0[m8:len(geo8_1)-1]

# moving the files selected into the test directory.

for f in tqdm(files1):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files2):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files3):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files4):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files5):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files6):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files7):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files8):
  shutil.move(f,'/content/actual_data/test')
  
# calculating how many files we have to take from each directory to consititute the train data for the class 1.

m1 = int(len(geo1_1)*train_ratio)
m2 = int(len(geo2_1)*train_ratio) 
m3 = int(len(geo3_1)*train_ratio) 
m4 = int(len(geo4_1)*train_ratio) 
m5 = int(len(geo5_1)*train_ratio) 
m6 = int(len(geo6_1)*train_ratio) 
m7 = int(len(geo7_1)*train_ratio)  
m8 = int(len(geo8_1)*train_ratio) 

# taking the first 70% from each GEO image for class 1.

files1 = geo1_1[:m1]
files2 = geo2_1[:m2]
files3 = geo3_1[:m3]
files4 = geo4_1[:m4]
files5 = geo5_1[:m5]
files6 = geo6_1[:m6]
files7 = geo7_1[:m7]
files8 = geo8_1[:m8]

# moving the files selected into the train/1 directory.

for f in tqdm(files1):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files2):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files3):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files4):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files5):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files6):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files7):
  shutil.move(f,'/content/actual_data/train/1')
for f in tqdm(files8):
  shutil.move(f,'/content/actual_data/train/1')


print('*'*100)

# taking the reamining 30% from each GEO image for class 1.

files1 = geo1_1[m1:]
files2 = geo2_1[m2:]
files3 = geo3_1[m3:]
files4 = geo4_1[m4:]
files5 = geo5_1[m5:]
files6 = geo6_1[m6:]
files7 = geo7_1[m7:]
files8 = geo8_1[m8:]

# moving the files selected into the test directory.

for f in tqdm(files1):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files2):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files3):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files4):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files5):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files6):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files7):
  shutil.move(f,'/content/actual_data/test')
for f in tqdm(files8):
  shutil.move(f,'/content/actual_data/test')
  
# zipping the resultant dataset

!zip -r /content/actual_data.zip /content/actual_data