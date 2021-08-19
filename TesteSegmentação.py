# TESTE_SIGATOKA
"""
Created on Tue Aug 10 21:15:32 2021

@author: Italla
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
# TESTE_SIGATOKA
# Tentar diferentes espaços de cores para encontrar o melhor canal de cores.
image = cv2.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folha.jpg')
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Canal RGB 
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]
fig, ax = plt.subplots(1,3, figsize=(15,15))
ax[0].imshow(R)
ax[1].imshow(G)
ax[2].imshow(B)

# Lab Color Space - https://en.wikipedia.org/wiki/CIELAB_color_space
L = img_lab[:, :, 0]
a = img_lab[:, :, 1]
b = img_lab[:, :, 2]
fig, ax = plt.subplots(1,3, figsize=(15,15))
ax[0].imshow(L)
ax[1].imshow(a)
ax[2].imshow(b)

# HSV Color Space - https://en.wikipedia.org/wiki/HSL_and_HSV
H = img_hsv[:, :, 0]
S = img_hsv[:, :, 1]
V = img_hsv[:, :, 2]
fig, ax = plt.subplots(1,3, figsize=(15,15))
ax[0].imshow(H)
ax[1].imshow(S)
ax[2].imshow(V)

# K-means clustering in opencv - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
pixel_vals = b.flatten()
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Uma vez que estamos interessados ​​apenas em pixels de folha reais, escolhemos 2 clusters 
# um cluster para pixels de folha reais e outro para pixels de fundo indesejados.
K = 2
retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((b.shape))
pixel_labels = labels.reshape(img_lab.shape[0], img_lab.shape[1])
# exibindo imagem segmentada 
plt.imshow(segmented_image)

# Fazendo isso, alguns pixels indesejados que estão agrupados no cluster principal podem ser evitados. 
# Ref - https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac2718a64ade63475425558aa669a943a
pixel_labels = np.uint8(pixel_labels)
ret, components = cv2.connectedComponents(pixel_labels, connectivity=8)
plt.imshow(components, cmap='gray')

indices = []
for i in range(1, ret):
    row, col = np.where(components==i)
    indices.append(max(len(row), len(col)))
component = np.argmax(np.array(indices))
main_component = component+1   # indexação começa em 0, portanto, incrementamos em 1 para obter o índice de componente real
# criando uma máscara e extraindo pixels correspondentes ao cluster ao qual a folha pertence. 
# 1 para pixels de folha reais e 0 para outros pixels
mask = np.where(components==main_component, 1, 0)
B = image[:, :, 0]
G = image[:, :, 1]
R = image[:, :, 2]
# Extraia apenas pixels mascarados
r = R*mask
g = G*mask
b = B*mask
final_img = np.dstack((r, g, b))
plt.imshow(final_img)

v_sum = np.sum(mask, axis=0)
h_sum = np.sum(mask, axis=1)
w = np.count_nonzero(v_sum)
h = np.count_nonzero(h_sum)
x_indices = np.where(v_sum != 0)
y_indices = np.where(h_sum != 0)
x = x_indices[0][0]
y = y_indices[0][0]
final_crop_img = final_img[y:y+h-1, x:x+w-1, :]
final_crop_img = np.uint8(final_crop_img)
img = cv2.resize(final_crop_img, (640,480), interpolation=cv2.INTER_AREA)
plt.imshow(img)

##################################################

import keras
from keras import regularizers
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
warnings.filterwarnings('ignore')
import itertools
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

path = '/My Drive/Leaf Disease Classification/'
dataset = '/My Drive/Leaf Disease Classification/Soyabean-Original/'
folders = os.listdir(dataset)
folders.sort()
# Armazenando dados de imagem em arquivo csv 
img_names = []
img_array = []
labels = []
for folder in folders:
    imgs = os.listdir(dataset+folder+os.sep)
    imgs.sort()
    for img in imgs:
        im = cv2.imread(dataset+folder+os.sep+img)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_size = cv2.resize(im_rgb, (144, 144), interpolation=cv2.INTER_AREA)
        img_names.append(img)
        img_array.append(im_size)
        labels.append(folder)
df = pd.DataFrame({'Image ID': img_names, 'Class': labels})
df.to_csv(path+'Files-Original/soyabean_original.csv', index=False)
# Salvando dados de imagem no formato de matriz numpy 
X = np.asarray(img_array)
np.save(path+'Files-Original/imgs_original', X)

data = pd.read_csv(path+'Files-Original/soyabean_original.csv')
data.head()

# Mapeando rótulos de classe originais para valores inteiros 
# Ex: Bacterial Blight-0, Brown Spot-1, ..etc 
labels = {}
val_cnt = 0
for folder in folders:
    labels[folder] = val_cnt
    val_cnt = val_cnt+1
X = np.load(path+'Files-Original/imgs_original.npy')
y = data['Class'].map(labels).values
data['Image Array'] = X.tolist()
data['Labels'] = y
data.head()

X = data.drop(['Image ID', 'Class', 'Labels'], axis=1)
y = data['Labels']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=0)
imbalance_train = X_tr.copy()
imbalance_train['Labels'] = y_tr
imbalance_train['Labels'].value_counts()

# Referência: upsampling em python-https: //elitedatascience.com/imbalanced-classes
majority_class = imbalance_train[imbalance_train.Labels == 6]
# Upsampling para dataset desequilíbrio em Python 
upsampled_classes = [majority_class]
minority_labels = [0,1,2,3,4,5,7,8,9]
for i in minority_labels:
    minority_class = imbalance_train[imbalance_train.Labels == i]
    minority_upsampled = resample(minority_class, replace = True,     n_samples = majority_class.shape[0],random_state = 0)
    upsampled_classes.append(minority_upsampled)
train_upsampled = pd.concat(upsampled_classes)
train_shuffled = train_upsampled.sample(frac=1, random_state=0)
X_tr1 = train_shuffled['Image Array'].values
y_tr1 = train_shuffled['Labels'].values
y_train = keras.utils.to_categorical(y_tr1, num_classes=10)
y_test = keras.utils.to_categorical(y_te, num_classes=10)
train_X = X_tr1.tolist()
X_train = np.asarray(train_X)
X_te1 = X_te['Image Array'].values
test_X = X_te1.tolist()
X_test = np.asarray(test_X)
# Dimensionando valores de pixel - Normalização 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

# Executando Aumento de Dados em dados de treinamento 
datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
datagen.fit(X_train)
it_train = datagen.flow(X_train, y_train, batch_size=4)
steps = int(X_train.shape[0]/4)

# Parâmetros do modelo
input_shape = (144, 144, 3)
num_classes = 10
# Salvando pesos do modelo e outros parâmetros importantes
checkpoint = ModelCheckpoint(path+"Results/model_complex.h5", monitor="val_acc",mode="max", save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=path+"Results/graph_complex", histogram_freq=4, batch_size=4,update_freq='epoch')
callbacks = [checkpoint, tensorboard]
# model architecture 
model1 = Sequential()
model1.add(Conv2D(32,(3,3),activation='relu',padding='same',strides=(2,2),kernel_initializer='he_normal',input_shape=input_shape))
model1.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model1.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model1.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(BatchNormalization())
model1.add(Dropout(0.3))
model1.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model1.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(BatchNormalization())
model1.add(Dropout(0.3))
model1.add(Flatten())
model1.add(Dense(512,activation='relu',kernel_initializer='he_normal'))
model1.add(Dropout(0.5))
model1.add(Dense(128,activation='relu',kernel_initializer='he_normal'))
model1.add(Dropout(0.5))
model1.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
model1.add(Dense(num_classes, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model1.summary()
# ajustar o modelo nos dados de treinamento. 
history = model1.fit_generator(it_train,steps_per_epoch=steps,epochs=300, validation_data=(X_test, y_test),verbose=1, callbacks=callbacks)
sns.set()
plt.plot(history.history['acc'], 'r')
plt.plot(history.history['val_acc'], 'b')
plt.legend({'Train Accuracy': 'r', 'Test Accuracy':'b'})
plt.show()
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.legend({'Train Loss': 'r', 'Test Loss':'b'})
plt.show()

model1.load_weights(path+'Results/Result-Original/model_complex.h5')
y_pred_tr = model1.predict(X_train)
y_tr2 = y_train.argmax(1)
y_pred_tr = y_pred_tr.argmax(1)
train_acc = accuracy_score(y_tr2, y_pred_tr)
y_pred = model1.predict(X_test)
y_te2 = y_test.argmax(1)
y_pred = y_pred.argmax(1)
test_acc = accuracy_score(y_te2, y_pred)
print("Train Accuracy : ", train_acc)
print("Test Accuracy : ", test_acc)

# model architecture
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', padding='same',strides=(2, 2), kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2, 2), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same', strides=(1, 1), kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
# modelo de ajuste no 
history = model.fit_generator(it_train,steps_per_epoch=steps, epochs=250,validation_data=(X_test, y_test),verbose=1, callbacks=callbacks)
sns.set()
plt.plot(history.history['acc'], 'r')
plt.plot(history.history['val_acc'], 'b')
plt.legend({'Train Accuracy': 'r', 'Test Accuracy':'b'})
plt.show()
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.legend({'Train Loss': 'r', 'Test Loss':'b'})
plt.show()

model1.load_weights(path+'Results/Results-Original/model_11.h5')
y_pred_tr = model1.predict(X_train)
y_tr2 = y_train.argmax(1)
y_pred_tr = y_pred_tr.argmax(1)
train_acc = accuracy_score(y_tr2, y_pred_tr)
y_pred = model1.predict(X_test)
y_te2 = y_test.argmax(1)
y_pred = y_pred.argmax(1)
test_acc = accuracy_score(y_te2, y_pred)
print("Train Accuracy : ", train_acc)
print("Test Accuracy : ", test_acc)
