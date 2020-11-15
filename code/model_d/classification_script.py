import numpy as np
import pickle
import os
import librosa
#from scipy.misc import imread
from sklearn.metrics import classification_report,confusion_matrix
import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
#import theano 
from keras.models import Model
#from keras import backend as K
#K.set_image_dim_ordering('th')
from random import shuffle
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
#from spp.SpatialPyramidPooling import SpatialPyramidPooling
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

weightpath_name='fusion_weights.hdf5'

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


x_test = np.load('/users/home/s18023/DCASE2020/taskb/fold1/x_test.npy')
labels_test = np.load('/users/home/s18023/DCASE2020/taskb/fold1/y_test.npy')

labels_test = keras.utils.to_categorical(labels_test, num_classes)

fs=2000

model1 = models.load_model('/users/home/s18023/DCASE2020/taskb/final/model_d/model_a.h5py')
model2 = models.load_model('/users/home/s18023/DCASE2020/taskb/final/model_d/model_c.h5py')
#model2 = models.load_model('/users/home/s18023/DCASE2020/taskb/fold1/best_Model_gap_50k_retrain.h5py')
first_model = model1.layers[12].output
first_model = models.Model(inputs=model1.input, outputs=first_model)
second_model = model2.layers[9].output
second_model = models.Model(inputs=model2.input, outputs=second_model)
first_model.summary()
second_model.summary()
pred1_train = first_model.predict(x_train)
pred1_test = first_model.predict(x_test)
pred2_train = second_model.predict(x_train)
pred2_test = second_model.predict(x_test)

x_train = np.column_stack((pred1_train, pred2_train))
x_test = np.column_stack((pred1_test, pred2_test))
print(x_train.shape)
print(x_test.shape)

input_audio = layers.Input(shape=(x_train.shape[1],))
model = layers.Dense(32, activation='relu')(input_audio)
#model = layers.Dense(32, activation='relu')(model)
model = layers.Dense(3, activation='softmax')(model)
model = Model(input_audio, model)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()

path_save=os.path.join("/users/home/s18023/DCASE2020/taskb/final/model_d")
os.chdir(path_save)

checkpointer = ModelCheckpoint(filepath=weightpath_name,monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)
hist=model.fit(x_train, y_train,batch_size=32,epochs=50,verbose=1, validation_split=0.2,callbacks=[checkpointer] ) #v



model.load_weights(weightpath_name)
model.save('fusion_model.h5py')
model.summary()
score=model.predict([x_test, x_test]);
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
#print("Accuracy:",metrics.accuracy_score(y_test_1, y_class)*100,"%")
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
