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

#%%
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
#fs=160000
model =Sequential()
model.add(Conv1D(16,64,strides=2,input_shape=(fs,1), name='layer1')) #layer1
model.add(BatchNormalization(name='layer2'))
convout1= Activation('relu', name='layer3')
model.add(convout1) 
model.add(AveragePooling1D(pool_size=2, padding='valid', name='layer4')) 
model.add(Conv1D(32,32,strides=2, name='layer5')) 
model.add(BatchNormalization(name='layer6'))
convout1= Activation('relu', name='layer7')
model.add(convout1) 
model.add(AveragePooling1D(pool_size=2, padding='valid', name='layer8'))
#model.add(GlobalAveragePooling1D()) 
model.add(Conv1D(64,16,strides=2, name='layer9'))
model.add(ZeroPadding1D(padding=16, name='layer10'))
model.add(BatchNormalization(name='layer11'))
convout1= Activation('relu', name='layer12')
model.add(convout1)
#model.add(AveragePooling1D(pool_size=2, padding='valid'))
model.add(GlobalAveragePooling1D(name='layer13')) 
#model.add(Flatten())
#model.add((Dense(64)))
#model.add((Activation('relu')))
model.add(Dense(32, name='layer14'))
model.add(Activation('relu', name='layer15'))
model.add(Dropout(0.3, name='layer16'))
model.add(Dense(3, name='layer17'))
model.add(Activation('softmax', name='layer18'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()


path_save=os.path.join("/users/home/s18023/DCASE2020/taskb/final/model_a")
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

pred_prob1=(np.asarray(pred_prob))/80

y_class= pred_prob1.argmax(axis=-1)

actual_labels=(labels_test[0:np.shape(x_test)[0]:split_fac])
#print(actual_labels.shape)
#print(y_class.shape)
#print(np.argmax(actual_labels, axis=1).shape)
#print(np.max(np.argmax(actual_labels, axis=1)))
#print(np.unique(np.argmax(actual_labels, axis=1)))
#print(np.unique(y_class))
dirs=['indoor', 'transportation', 'outdoor']
target_names = dirs
#print(classification_report(actual_labels,y_class,target_names=target_names))
asd=(confusion_matrix(np.argmax(actual_labels, axis=1),y_class))
accur=np.trace(asd)/np.sum(asd)
print("Accuracy:", accur*100)
print("Log Loss:", log_loss(np.argmax(actual_labels, axis=1),pred_prob1))
