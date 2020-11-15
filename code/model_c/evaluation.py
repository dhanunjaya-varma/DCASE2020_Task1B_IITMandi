import numpy as np
import pickle
import os
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
from keras.models import Model
from random import shuffle
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.layers import LSTM,GlobalAveragePooling1D
from keras.models import load_model

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras import models
from keras import layers
from sklearn import metrics
import csv

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

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=16000, mono=True, duration=10.0)

    return sound_sample, sr

#x_test = np.load('/users/home/s18023/DCASE2020/taskb/fold1/x_test.npy')
#labels_test = np.load('/users/home/s18023/DCASE2020/taskb/fold1/y_test.npy')

#labels_test = keras.utils.to_categorical(labels_test, num_classes)

header = ['filename', 'scene_label', 'indoor', 'outdoor', 'transportation']
csvfile = open('/users/home/s18023/DCASE2020/taskb/final/model_c/evaluation_model_c.csv','w', newline='')
obj = csv.writer(csvfile, delimiter='\t')
obj.writerow(header)

fs=2000

model = models.load_model('/users/home/s18023/DCASE2020/taskb/final/model_c/model.h5py')

#model.summary()

fs=2000
#ex_test=[]
#labels_test=[]
#cl_test=0
test_data_path="/users/home/s18023/DCASE2020/evaluation/audio"
num_segment = split_fac;
#pred_prob = []

parts = os.listdir(test_data_path)
print(len(parts))
parts.sort()
for part in parts:
	pl = []
	pl.append(str(part))
	sound_sample,sr =load_audio(test_data_path+'/'+part)
	sound_sample *= 256
	example=np.array(sound_sample)
	sd=np.split(example,split_fac)
	sd = np.array(sd).reshape((80, 2000, 1))
	#print(sd.shape)
	score = model.predict(sd)
	pred = np.sum(score[0:0+num_segment,:],0)
	pred = pred/80
	if np.array(pred).argmax(axis=-1) == 0:
		pl.append('transportation')
	elif np.array(pred).argmax(axis=-1) == 1:
		pl.append('indoor')
	elif np.array(pred).argmax(axis=-1) == 2:
		pl.append('outdoor')
	pl.append(pred[1])
	pl.append(pred[2])
	pl.append(pred[0])
	obj.writerow(pl)
	#pl.append(np.array(pred).argmax(axis=-1))
	#pl.append(cl_test)
	#print(*pl, sep='\t')
	#pred_prob.append(pred)
	#labels_test=np.hstack((labels_test,np.tile(cl_test,split_fac)))
	#sum += 1
	#q=q+1
	#cl_test += 1

csvfile.close()
