
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D,regularizers
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
#import theano 

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
#K.set_image_dim_ordering('th')
from random import shuffle
from keras.callbacks import ModelCheckpoint
import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.layers import LSTM,GlobalAveragePooling1D
from keras.models import load_model
import scipy.io
import pickle
import os
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


from scipy.stats import mode
#import theano 
import matplotlib.pyplot as plt
import numpy as np

#K.set_image_dim_ordering('th')
from random import shuffle

import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import random 
from random import randint
from sklearn.svm import SVC
#%%
from sklearn.metrics import log_loss
from keras import models
from keras import layers
from sklearn import metrics

cl = 0.
sum = 0
q = 0
j=0
labels=[]
ex=[]
ex_test=[]
#labels_test=[]
p=0
h=0
cl_test=0
num_classes=3
#%%

split_fac=80;
time_analysis=1

weightpath_name='weights.hdf5'

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=16000, mono=True, duration=10.0)

    return sound_sample, sr

#%% data read

train_data_path="/users/home/s18023/DCASE2020/taskb/fold1/train"
for root, dirs, files in os.walk(train_data_path, topdown=False):
    for name in dirs:
        parts = []
        parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.wav')]
        print(name, "...")
        
        for part in parts:
                #image=color.rgb2gray(io.imread(os.path.join(root,name,part))
            #img = Image.open(os.path.join(root,name,part))
            sound_sample,sr =load_audio(os.path.join(root,name,part))
            sound_sample *= 256         
            example=np.array(sound_sample)
            sd=np.split(example,split_fac)
            ex.append(sd)
            labels=np.hstack((labels,np.tile(cl,split_fac)))
            sum += 1                        
            j=j+1
        cl += 1

                
print('X_data shape:', np.array(ex).shape)
print('labels_shape:', np.array(labels).shape)

#%%.................................................................................................

x_train=np.asarray(np.reshape(ex,(np.shape(ex)[0]*split_fac,fs,1)))
y_train=np.asarray(labels)
x_train=x_train.astype('float32')
y_train = keras.utils.to_categorical(y_train, num_classes)
np.save('/users/home/s18023/DCASE2020/taskb/fold1/x_train.npy',np.array(ex))
np.save('/users/home/s18023/DCASE2020/taskb/fold1/y_train.npy',np.array(labels))

#%% test data read...................................................................................
fs=2000
ex_test=[]
labels_test=[]
cl_test=0
test_data_path="/users/home/s18023/DCASE2020/taskb/fold1/test"

for root, dirs, files in os.walk(test_data_path, topdown=False):
    for name in dirs:
        parts = []
        parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.wav')]
        print(name, "...")
        
        for part in parts:
                #image=color.rgb2gray(io.imread(os.path.join(root,name,part))
            #img = Image.open(os.path.join(root,name,part))
            sound_sample,sr =load_audio(os.path.join(root,name,part))
            sound_sample *= 256         
            example=np.array(sound_sample)
            sd=np.split(example,split_fac)
            ex_test.append(sd)
            labels_test=np.hstack((labels_test,np.tile(cl_test,split_fac)))
            sum += 1                        
            q=q+1
        cl_test += 1

                
print('X_data_test shape:', np.array(ex_test).shape)

print('labels_test_shape:', np.array(labels_test).shape)

x_test=np.asarray(np.reshape(ex_test,(np.shape(ex_test)[0]*split_fac,fs,1)))
y_test=np.asarray(labels_test)
x_test=x_test.astype('float32')
y_test = keras.utils.to_categorical(y_test, num_classes)


np.save('/users/home/s18023/DCASE2020/taskb/fold1/x_test.npy',x_test)
np.save('/users/home/s18023/DCASE2020/taskb/fold1/y_test.npy',labels_test)

x_train = np.load('/users/home/s18023/DCASE2020/taskb/fold1/x_train.npy')
y_train = np.load('/users/home/s18023/DCASE2020/taskb/fold1/y_train.npy')

x_test = np.load('/users/home/s18023/DCASE2020/taskb/fold1/x_test.npy')
labels_test = np.load('/users/home/s18023/DCASE2020/taskb/fold1/y_test.npy')

x_train = x_train.reshape((734800, 2000, 1))
#x_test = x_test.reshape((334800, 2000))
y_train = keras.utils.to_categorical(y_train, num_classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes)

fs=2000

param_G=np.load("/users/home/s18023/DCASE2020/taskb/sound8.npy", encoding = 'latin1',allow_pickle=True).item()
initia_weights1=[np.reshape(param_G['conv1']['weights'],(64,1,16)),param_G['conv1']['biases'],param_G['conv1']['gamma'],param_G['conv1']['beta'],param_G['conv1']['mean'],param_G['conv1']['var'],np.reshape(param_G['conv2']['weights'],(32,16,32)),param_G['conv2']['biases'],param_G['conv2']['gamma'],param_G['conv2']['beta'],param_G['conv2']['mean'],param_G['conv2']['var']]#,np.reshape(param_G['conv3']['weights'],(16,32,64)),param_G['conv3']['biases'],param_G['conv3']['gamma'],param_G['conv3']['beta'],param_G['conv3']['mean'],param_G['conv3']['var']]#,np.reshape(param_G['conv4']['weights'],(8,64,128)),param_G['conv4']['biases'],param_G['conv4']['gamma'],param_G['conv4']['beta'],param_G['conv4']['mean'],param_G['conv4']['var']]#,np.reshape(param_G['conv5']['weights'],(4,128,256)),param_G['conv5']['biases'],param_G['conv5']['gamma'],param_G['conv5']['beta'],param_G['conv5']['mean'],param_G['conv5']['var'],np.reshape(param_G['conv6']['weights'],(4,256,512)),param_G['conv6']['biases'],param_G['conv6']['gamma'],param_G['conv6']['beta'],param_G['conv6']['mean'],param_G['conv6']['var'],np.reshape(param_G['conv7']['weights'],(4,512,1024)),param_G['conv7']['biases'],param_G['conv7']['gamma'],param_G['conv7']['beta'],param_G['conv7']['mean'],param_G['conv7']['var']]#,np.reshape(param_G['conv8']['weights'],(8,1024,1000)),param_G['conv8']['biases'],np.reshape(param_G['conv8_2']['weights'],(8,1024,401)),param_G['conv8_2']['biases']]


#%% note that 0 is normal and 1 is abnormal in log mel features.....TNR is the measure of abnormals are classififed as abnormal...
model =Sequential()

model.add(Conv1D(16,64,strides=2,input_shape=(2000,1))) #layer1
model.add(ZeroPadding1D(padding=16))
model.add(BatchNormalization()) #layer2
convout1= Activation('relu')
model.add(convout1) #layer3
model.add(MaxPooling1D(pool_size=8, padding='valid')) #layer4
model.add(Conv1D(32,32,strides=2)) #layer5
model.add(ZeroPadding1D(padding=8))
model.add(BatchNormalization()) #layer6
convout2= Activation('relu')
model.add(convout2) #layer7
#model.add(Dropout(0.5))
model.set_weights(initia_weights1)
model.add(GlobalAveragePooling1D())
#model.add((Dense(128)))
#model.add((Activation('relu')))
model.add((Dense(64)))
model.add((Activation('relu')))
#model.add((Dense(32)))
#model.add((Activation('relu')))
#model.add(Dropout(0.3))
model.add(Dense(3))
model.add(Activation('softmax'))



#%%

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

path_save=os.path.join("/users/home/s18023/DCASE2020/taskb/final/model_c")
os.chdir(path_save)

checkpointer = ModelCheckpoint(filepath=weightpath_name,monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)
hist=model.fit(x_train, y_train,batch_size=32,epochs=50,verbose=1, validation_split=0.2,callbacks=[checkpointer] ) #v



model.load_weights(weightpath_name)
model.save('model.h5py')
score=model.predict(x_test);
t=0;
num_segment=split_fac;
pred_prob=[]
for i in range(int(np.shape(x_test)[0]/split_fac)):
    
    pred=np.sum(score[t:t+num_segment,:],0)
    pred_prob.append(pred)
    t=t+num_segment
#

pred_prob=(np.asarray(pred_prob))/80

y_class= pred_prob.argmax(axis=-1)

actual_labels=(labels_test[0:np.shape(x_test)[0]:split_fac])
dirs=['indoor', 'transportation', 'outdoor']
target_names = dirs
#print(classification_report(actual_labels,y_class,target_names=target_names))
asd=(confusion_matrix(np.argmax(actual_labels, axis=1),y_class))
accur=np.trace(asd)/np.sum(asd)
print("Accuracy:", accur*100)
cm = asd.astype('float') / asd.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())
y_test_1 = np.argmax(actual_labels, axis=1)
print("Accuracy:",metrics.accuracy_score(y_test_1, y_class)*100,"%")
print("Log Loss:", log_loss(y_test_1, pred_prob))

for i in range(3):
    print(i)
    idx_label = np.where(y_test_1 == i)
    y = []
    p = []
    print(len(idx_label[0]))
    for j in range(len(idx_label[0])):
        y.append(y_test_1[idx_label[0][j]])
        p.append(pred_prob[idx_label[0][j]])
    y = np.array(y)
    p = np.array(p)
    print("Log Loss:", log_loss(y, p, labels=list(range(3))))
