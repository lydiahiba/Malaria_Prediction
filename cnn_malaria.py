#data preprocessing
import pandas as pd
#math operations
import numpy as np
#machine learning
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
            
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
import random


PARA_DIR = "/content/cell_images/cell_images/Parasitized/"
UNIF_DIR =  "/content/cell_images/cell_images/Uninfected/"

import os
Pimages = os.listdir(PARA_DIR)
Nimages = os.listdir(UNIF_DIR)


sample_normal = random.sample(Nimages,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('/content/cell_images/cell_images/Uninfected/'+sample_normal[1])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Uninfected')
plt.show()

PARA_DIR = "/content/cell_images/cell_images/Parasitized/"
UNIF_DIR =  "/content/cell_images/cell_images/Uninfected/"

sample_normal = random.sample(Pimages,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('/content/cell_images/cell_images/Parasitized/'+sample_normal[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Parasitized')
plt.show()


data=[]
labels=[]

#Parasitized=os.listdir("/content/cell_images/cell_images/Parasitized/")
#Uninfected=os.listdir("/content/cell_images/cell_images/Uninfected/")


for a in Parasitized:
    try:
        image=cv2.imread("/content/cell_images/cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((75,75))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")


for b in Uninfected:
    try:
        image=cv2.imread("/content/cell_images/cell_images/Uninfected/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((75, 75))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")


Cells=np.array(data)
labels=np.array(labels)
np.save("Cells",Cells)
np.save("labels",labels)

Cells=np.load("Cells.npy")
labels=np.load("labels.npy")

s=np.arange(Cells.shape[0])
np.random.shuffle(s)

Cells=Cells[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(Cells)
len_data

x_train=Cells[(int)(0.1*len_data):]
x_test=Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)
x_train.shape

(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]

#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed
np.random.seed(0)

import tensorflow as tf

base_model = tf.keras.applications.InceptionV3(input_shape=(75,75,3),
                                               include_top=False,
                                               weights = "imagenet"
                                               )

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation="softmax")
])

initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=6000,
    decay_rate=0.90,
    staircase=True)

# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy()])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint('weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5', monitor='val_loss', save_best_only=True)]


x_train.shape
y_train.shape   
x_test.shape
y_test.shape

#Fit the model with min batch size as 32 can tune batch size to some factor of 2^power ] 
h=model.fit(x_train,y_train,batch_size=32,callbacks=callbacks, validation_data=(x_test,y_test),epochs=10,verbose=1)

import h5py

from numpy import loadtxt
from keras.models import load_model
model = load_model('.hdf5')

score=model.evaluate(x_test,y_test)
print(score)

accuracy = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])

from sklearn.metrics import confusion_matrix
pred = model.predict(x_test)
pred = np.argmax(pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)

CM = confusion_matrix(y_true, pred)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
plt.show()

i=9
pred = model.predict(x_test,batch_size=1)
pred = np.argmax(pred,axis = 1)


plt.plot(h.history['accuracy'])
plt.plot(h.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Epochs")
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


import numpy as np
from sklearn.metrics import auc, roc_curve
fpr_keras, tpr_keras, thresholds = roc_curve(y_true.ravel(), pred.ravel())
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    plot_roc_curve(fpr_keras, tpr_keras)

    from sklearn.metrics import classification_report



print('{}'.format(classification_report(y_true , pred)))

# get predictions on the test set
y_hat = model.predict(x_test)

# define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
malaria_labels = ['Parasitized','Uninfected']

# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=12, replace=False)):
    ax = fig.add_subplot(4,4, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(malaria_labels[pred_idx], malaria_labels[true_idx]),
                 color=("blue" if pred_idx == true_idx else "orange"))