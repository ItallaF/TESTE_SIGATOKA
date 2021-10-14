# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:19:48 2021

@author: Italla
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#-------------Imagens de Teste-------------------------------------
#original_image = cv2.imread('img/16/40e9d306-b56d-4066-b081-895eb2cfed1f.jpg')
#original_image = cv2.imread('img/16/06b4eeba-8283-46a8-9582-4fc8fecec4c6.jpg')
#original_image = cv2.imread('img/37/2bbc7e78-bdea-42c9-aca7-3f37610c95d9.jpg')
#original_image = cv2.imread('img/entrada/folha-de-mamao-menor.jpg')
#original_image = cv2.imread('img/entrada/sigatoka.jpg')
#original_image = cv2.imread('img/entrada/sigatoka1.jpeg')
#original_image = cv2.imread('img/entrada/sigatoka4.jpg')
#original_image = cv2.imread('img/entrada/sigatoka3.jpeg')
#original_image = cv2.imread('img/entrada/tomate1.jpg')
original_image = cv2.imread("C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/FolhamamaoSemFundo.jpg")
#original_image = cv2.imread("img/entrada/plantacao-de-bananeira-png-4.jpg")
#original_image = cv2.imread("img/entrada/folha1.jpg")
#original_image = cv2.imread("img/entrada/folha3.jpg")
#original_image = cv2.imread("img/saida/plantacao-de-bananeira-png-3.jpg")

#-------------Fazendo Converções de Cores-------------------------------------
image_YCrCb = cv2.cvtColor(original_image, cv2.COLOR_YCrCb2BGR)
image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_yiq = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
'''image_HSV = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
image_LAB = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
image_HLS = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)
cv2.imshow("original_image",original_image)
cv2.imshow("image_YCrCb",image_YCrCb)
cv2.imshow("image_rgb",image_rgb)
cv2.imshow("image_HSV",image_HSV)
cv2.imshow("image_LAB",image_LAB)
cv2.imshow("image_HLS",image_HLS)'''
#NOTA: sera trabalhado nos canais de cores YCrCb e RGB, endo o YCrCb para indentificação é RGB para controle

#-------------Aplicando o K-means-------------------------------------
#-------------Aplicando no K-pdrão de cor YCrCb-------------------------------------
img = image_YCrCb
Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res_YCrCb = center[label.flatten()]
res2_YCrCb = res_YCrCb.reshape((img.shape))
cv2.imshow('res2_YCrCb',res2_YCrCb)
#-------------Aplicando no K-pdrão de cor iRGB-------------------------------------
img = image_rgb
Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res_rgb = center[label.flatten()]
res2_rgb = res_rgb.reshape((img.shape))
cv2.imshow('res2_rgb',res2_rgb)

#--------------------YIQ-----------------------------------
img = image_yiq
w, h ,c= img.shape
print(img.shape)
Y = np.zeros((w, h))
I = np.zeros((w, h))
Q = np.zeros((w, h))
for i in range(0, w):
    for j in range(0, h):
        R = img[i, j, 2]
        G = img[i, j, 1]
        B = img[i, j, 0]
        # RGB -> YIQ
        Y[i,j] = int((0.299*R) + (0.587 * G) + (0.114 * B))
        I[i,j] = int((0.596 * R) - (0.274 * G) - (0.322 * B))
        Q[i,j] = int((0.211 * R) - (0.523 * G) + (0.312 * B))
img_out = cv2.merge((Y, I, Q))
res_yiq = img_out.astype(np.uint8)

imag = res_yiq
canal1 = imag[:, :, 0]
canal2 = imag[:, :, 1]
canal3 = imag[:, :, 2]

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res_yiq = res_yiq.reshape((img.shape))
cv2.imshow('res_yiq',res_yiq)


#-------------Aplicando mascara-------------------------------------
gray = cv2.cvtColor(res_yiq, cv2.COLOR_BGR2GRAY)

img_hsv_gaussian = cv2.GaussianBlur (gray, (5,5), 0)
ret3, otsu1 = cv2.threshold (img_hsv_gaussian, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("binarização",otsu1)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1) #"mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
masked = cv2.bitwise_and(image, image, mask=otsu1) #"image", image, mask=mask
cv2.imshow("masked",masked)

'''imag = YCrCb
canal1 = imag[:, :, 0]
canal2 = imag[:, :, 1]
canal3 = imag[:, :, 2]
cv2.imshow("canal1",canal1)
cv2.imshow("canal2",canal2)
cv2.imshow("canal3",canal3)'''
cv2.imshow("original_image",original_image)
cv2.imshow("image_YCrCb",image_YCrCb)
cv2.imshow("image_rgb",image_rgb)
cv2.imshow("image_yiq",image_yiq)
cv2.waitKey(0)

#imgRGB = cv.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/IMAGENS FORNECIDAS/DSC01030.jpg')

#---------------------------Calculo da imagem-----------------------------
print("número de pixels na planta:",len(masked.nonzero()[0])*100)
# distância é 50
distance_top=1000
Area=(pow((0.000122*(distance_top-0.304)/0.304),2)*len(masked.nonzero()[0]))
print("folha area:",round(Area, 2))

res_yiq = np.asarray(res_yiq, dtype=np.float32)/255
print("Dimensões da imagem ", res_yiq.shape)
plt.figure(figsize=(3, 3))
im = plt.imshow(res_yiq, aspect='auto')
plt.axis("off")
plt.show()

MTR = np.transpose(res_yiq[ :, :, 0])
MTG = np.transpose(res_yiq[ :, :, 1])
MTB = np.transpose(res_yiq[ :, :, 2])

MT = np.zeros((640, 480, 3))

MT[ :, :, 0 ], MT[ :, :, 1], MT[ :, :, 2] = MTR, MTG, MTB

plt.figure(figsize=(3, 3))
im = plt.imshow(res_yiq, aspect='auto')
plt.axis("off")
plt.show()

res_yiq[99:200, :, 0] = 0

plt.figure(figsize=(3, 3))
im = plt.imshow(res_yiq, aspect='auto')
plt.axis("off")
plt.show()
res_yiq[140:168, :, 0] = 0
res_yiq[140:168, :, 1] = 0
plt.figure(figsize=(3, 3))
im = plt.imshow(res_yiq, aspect='auto')
plt.axis("off")
plt.show()
