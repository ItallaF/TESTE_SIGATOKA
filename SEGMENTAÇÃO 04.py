# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 08:53:55 2021

@author: Italla
"""

from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage

#Segmentação baseada em região

image = plt.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folhaBananeira.jpg')
image.shape
plt.imshow(image)

#CONVERTENDO A IMAGEM EM CINZA
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')

print ("Dimensões da imagem ", gray.shape)

# A média dos valores de pixel é usado como limite 
# Valor do pixel menor que o limite, ele será plano de fundo
# Valor do pixel menor que o limite, ele será objeto
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')

#Definido diferentes limites para diferentes objetos
gray = rgb2gray(image)
gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')

##########################################################

#Segmentação de detecção de borda
image = plt.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/IMAGENS FORNECIDAS/DSC01030.jpg')
plt.imshow(image)

# convertendo para escala de cinza
gray = rgb2gray(image)

# definir o filtro sobel (horizontal e vertical) 
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(sobel_horizontal, 'é um kernel para detectar bordas horizontais')
 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(sobel_vertical, 'é um kernel para detectar bordas verticais')

# convolve este filtro sobre a imagem usando o convolve função do ndimage pacote de scipy .
out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
# o modo aqui determina como a matriz de entrada é estendida quando o filtro se sobrepõe a uma borda.

# mostrando o resultado
plt.imshow(out_h, cmap='gray')

plt.imshow(out_v, cmap='gray')

kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
print(kernel_laplace, 'é um kernel laplaciano')

# método detecta bordas horizontais e verticais
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
plt.imshow(out_l, cmap='gray')

###################################################################

#Segmentação de imagem baseada em clustering

# dividindo por 255 para trazer os valores de pixel entre 0 e 1
pic = plt.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/IMAGENS FORNECIDAS/DSC01030.jpg')/255  
print(pic.shape)
plt.imshow(pic)

# converte uma matriz tridimencional em uma matriz bidimensional (comprimento * largura, canais)
# A função cluster_centers_ de k-means retornará os centros do cluster
# A função labels_ retonará o rótulo para cada pixel (ela dirá qual pixel da imagem pertence a qual cluster).
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)

