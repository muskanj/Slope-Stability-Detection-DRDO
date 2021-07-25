!mkdir Data # creating a directory

!pip install tifffile
!pip install tensorflow-addons # installing the required packages 

# importing necessities 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
import shutil
import glob
import random
from tqdm import tqdm
import os

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils import to_categorical
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
import tensorflow as tf
import tensorflow.keras.layers as K
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization,Input, SeparableConv2D, Concatenate,ReLU, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam

# unzipping the data collected into the data folder created above
!unzip /content/drive/MyDrive/Final_train_test_data/actual_data.zip -d /content/Data

# function to read tif files using tifffile library
def read_img(path):
  img = tif.imread(path)
  return img

# defining the lists to store the train data and test data
train_data = []
train_labels = []
test_data = []
test_labels = []

# storing the filenames in each folder in lists
train_files_0 = glob.glob('/content/Data/content/actual_data/train/0/*tif')
train_files_1 = glob.glob('/content/Data/content/actual_data/train/1/*tif')

# depricated code
# val_files_0 = glob.glob('/content/Data/content/actual_data/val/0/*tif')
# val_files_1 = glob.glob('/content/Data/content/actual_data/val/1/*tif')
test_files = glob.glob('/content/Data/content/actual_data/test/*tif')

# taking the first 200k files for each class as the train data and 52k files for each class as the validation data
train_files = train_files_0 + train_files_1

# shuffling the train files
random.shuffle(train_files)

# loading the train data in the lists created above
for i in tqdm(train_files):
  im = read_img(i)
  train_data.append(im)
  train_labels.append(i[-5])

# loading the test data in the lists created above 
for i in tqdm(test_files):
  im = read_img(i)
  test_data.append(im)
  test_labels.append(i[-5])
  
# stacking the images to create the desired shape of the data so as to feed it to the network

train_data = np.stack(train_data)

test_data = np.stack(test_data)

# checking if the image is loaded correctly by checking its shape
print(train_data[1].shape)

print(train_data.shape)
print(test_data.shape)

plt.figure(figsize=(3,3))
plt.imshow((read_img(train_files[9988])/255.0),alpha=0.98,interpolation='nearest')
# plt.grid()






########### U-NET Architecture ###############

# defining the model
# using tensorflow-addons for various metrics
def def_model(inp_shape = (16,16,4)):
    model = keras.models.Sequential([
            Conv2D(32,(3,3),input_shape=inp_shape,strides=(1,1),padding='same',activation='relu'),
            Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),
        
            Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'),
            Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),
        
            Conv2D(128,(3,3),input_shape=inp_shape,strides=(1,1),padding='same',activation='relu'),
            Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),

            Conv2D(256,(3,3),input_shape=inp_shape,strides=(1,1),padding='same',activation='relu'),
            Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),
        
            Conv2D(128,(3,3),input_shape=inp_shape,strides=(1,1),padding='same',activation='relu'),
            Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),
        
            Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'),
            Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),
        
            Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu'),
            Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
            BatchNormalization(),

            Flatten(),
            Dense(256,activation='relu'),
            Dropout(0.5),
            Dense(2, activation="sigmoid")          
        ])
    model.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.Adam() ,metrics = ['accuracy',keras.metrics.AUC(name='auc'),tfa.metrics.F1Score(2),tfa.metrics.MatthewsCorrelationCoefficient(2)])
    return model

# loading and printing the summary of the model
model = def_model()
model.summary()

# deining the callbacks ReduceLROnPlateau to readuce learning rate with iterations
callback = keras.callbacks.ReduceLROnPlateau(
    monitor='loss',factor=0.25,patience=2, verbose=1,min_delta=0.0001,cooldown=0,min_lr=0.00001, mode='auto',
)

# definig the ModelCheckpoint so as to save the weights for the version of the train data
# when the next batch of the train data is used the model is loaded from these saved weights

check = tf.keras.callbacks.ModelCheckpoint(
    '/content/version_1_4Ltrain_104kval',
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,   
)

# converting the labels into integers because they are stored in the string format 
# the labels are stored in the lists defined

labels_train = []
labels_test = []

for i in train_labels:
  lab = int(i)
  labels_train.append(lab)
for i in test_labels:
  lab = int(i)
  labels_test.append(lab)
  
# one hot encoding of the labels using keras.utils.to_categorical

labels_train = tf.keras.utils.to_categorical(labels_train)
labels_test = tf.keras.utils.to_categorical(labels_test)

print(labels_train.shape)
print(labels_test.shape)

# fitting the data with the test data and the callbacks
model.fit(train_data,labels_train,
          validation_data=(test_data,labels_test),
          batch_size=128,
          epochs=100,
          callbacks=[callback,check]
)

# train data results
pred_train = np.argmax(model.predict(train_data,verbose=1),axis=1)
train_lab = np.argmax(labels_train,axis=1)
print('Accuracy :',accuracy_score(train_lab,pred_train))
print('F1 Score :',f1_score(train_lab,pred_train))
print('Matthews corrcoef :',matthews_corrcoef(train_lab,pred_train))
print('ROC AUC :',roc_auc_score(train_lab,pred_train))

print("*"*100)

# test data results
pred = np.argmax(model.predict(test_data,verbose=1),axis=1)
test_lab = np.argmax(labels_test,axis=1)
print('Accuracy :',accuracy_score(test_lab,pred))
print('F1 Score  :',f1_score(test_lab,pred))
print('Matthews corrcoef :',matthews_corrcoef(test_lab,pred))
print('ROC AUC  :',roc_auc_score(test_lab,pred))








############ Research paper smaller model ############

# defining the model
# using tensorflow-addons for various metrics
def def_model(inp_shape = (16,16,4)):
    model = keras.models.Sequential([
            Conv2D(64,(3,3),input_shape=inp_shape,strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
        
            Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),

            Flatten(),
            Dense(256,activation='relu'),
            Dropout(0.5),
            Dense(2, activation="sigmoid")          
        ])
    model.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.Adam() ,metrics = ['accuracy',keras.metrics.AUC(name='auc'),tfa.metrics.F1Score(2),tfa.metrics.MatthewsCorrelationCoefficient(2)])
    return model

model = def_model()
model.summary()

# # fitting the data with the test data and the callbacks
model.fit(train_data,labels_train,
          validation_data=(test_data,labels_test),
          batch_size=128,
          epochs=100,
          callbacks=[callback,check]
)

# train data results
pred_train = np.argmax(model.predict(train_data,verbose=1),axis=1)
train_lab = np.argmax(labels_train,axis=1)
print('Accuracy :',accuracy_score(train_lab,pred_train))
print('F1 Score :',f1_score(train_lab,pred_train))
print('Matthews corrcoef :',matthews_corrcoef(train_lab,pred_train))
print('ROC AUC :',roc_auc_score(train_lab,pred_train))

print("*"*100)

# test data results
pred = np.argmax(model.predict(test_data,verbose=1),axis=1)
test_lab = np.argmax(labels_test,axis=1)
print('Accuracy :',accuracy_score(test_lab,pred))
print('F1 Score  :',f1_score(test_lab,pred))
print('Matthews corrcoef :',matthews_corrcoef(test_lab,pred))
print('ROC AUC  :',roc_auc_score(test_lab,pred))








############## Research paper larger model #################


from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout
# defining the model
# using tensorflow-addons for various metrics
def def_model(inp_shape = (16,16,4)):
    model = keras.models.Sequential([
            Conv2D(64,(3,3),input_shape=inp_shape,strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),
        
            Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),

            Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(1,1)),

            Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),

            Flatten(),
            Dense(1024,activation='relu'),
            Dropout(0.5),
            Dense(2, activation="sigmoid")          
        ])
    model.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.Adam() ,metrics = ['accuracy',keras.metrics.AUC(name='auc'),tfa.metrics.F1Score(2),tfa.metrics.MatthewsCorrelationCoefficient(2)])
    return model

model = def_model()
model.summary()

# # fitting the data with the test data and the callbacks
model.fit(train_data,labels_train,
          validation_data=(test_data,labels_test),
          batch_size=128,
          epochs=100,
          callbacks=[callback,check]
)

# train data results
pred_train = np.argmax(model.predict(train_data,verbose=1),axis=1)
train_lab = np.argmax(labels_train,axis=1)
print('Accuracy :',accuracy_score(train_lab,pred_train))
print('F1 Score :',f1_score(train_lab,pred_train))
print('Matthews corrcoef :',matthews_corrcoef(train_lab,pred_train))
print('ROC AUC :',roc_auc_score(train_lab,pred_train))

print("*"*100)

# test data results
pred = np.argmax(model.predict(test_data,verbose=1),axis=1)
test_lab = np.argmax(labels_test,axis=1)
print('Accuracy :',accuracy_score(test_lab,pred))
print('F1 Score  :',f1_score(test_lab,pred))
print('Matthews corrcoef :',matthews_corrcoef(test_lab,pred))
print('ROC AUC  :',roc_auc_score(test_lab,pred))







######### Model mentioned in research papaer ..... ###############


def def_model2(inp_shape = (16,16,4)):
    input_img = Input(shape=inp_shape, name='ImageInput')
    x = Conv2D(8, (3,3), activation='relu', padding='same')(input_img)
    
    y = MaxPooling2D(8, (2,2), padding = 'same')(x)
    
    x = BatchNormalization()(y)
    x = ReLU()(x)
    x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
    x = x+y
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation='sigmoid')(x)
        
    model = tf.keras.Model(inputs=input_img, outputs=x)
    model.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.Adam() ,metrics = ['accuracy',keras.metrics.AUC(name='auc'),tfa.metrics.F1Score(2),tfa.metrics.MatthewsCorrelationCoefficient(2)])
    return model

# loadung and printing the summary of the model

model = def_model2()
model.summary()

# fitting the data with the test data and the callbacks
model.fit(train_data,labels_train,
          validation_data=(test_data,labels_test),
          batch_size=128,
          epochs=100,
          callbacks=[callback,check]
)

# train data results
pred_train = np.argmax(model.predict(train_data,verbose=1),axis=1)
train_lab = np.argmax(labels_train,axis=1)
print('Accuracy :',accuracy_score(train_lab,pred_train))
print('F1 Score :',f1_score(train_lab,pred_train))
print('Matthews corrcoef :',matthews_corrcoef(train_lab,pred_train))
print('ROC AUC :',roc_auc_score(train_lab,pred_train))

print("*"*100)

# test data results
pred = np.argmax(model.predict(test_data,verbose=1),axis=1)
test_lab = np.argmax(labels_test,axis=1)
print('Accuracy :',accuracy_score(test_lab,pred))
print('F1 Score :',f1_score(test_lab,pred))
print('Matthews corrcoef :',matthews_corrcoef(test_lab,pred))
print('ROC AUC :',roc_auc_score(test_lab,pred))