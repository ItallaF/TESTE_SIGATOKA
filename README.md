# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:05:18 2021

@author: aluno
"""

import numpy as np
import cv2
import urllib.request

img = cv2.imread('C:/Users/aluno/Desktop/Imagem em YCrCb.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
bin1 = cv2.adaptiveThreshold(suave, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
 21, 5)
resultado = np.vstack([
np.hstack([img, suave]),
np.hstack([bin1, bin2])
])
cv2.imshow("Binarização adaptativa da imagem", resultado)
cv2.waitKey(0)

import math
import pandas as pd
import mahotas
import numpy as np
import cv2
img = cv2.imread('C:/Users/aluno/Desktop/Imagem em YCrCb.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
T = mahotas.thresholding.otsu(suave)
temp = img.copy()
temp[temp > T] = 255
temp[temp < 255] = 0
temp = cv2.bitwise_not(temp)
T = mahotas.thresholding.rc(suave)
temp2 = img.copy()
temp2[temp2 > T] = 255
temp2[temp2 < 255] = 0
temp2 = cv2.bitwise_not(temp2)
resultado = np.vstack([np.hstack([img, suave]),np.hstack([temp, temp2])])
cv2.imshow("Binarização com método Otsu e Riddler-Calvard", resultado)
#hsvImage = cv2.cvtColor(resultado, cv2.COLOR_BGR2HSV)
#cv2.imshow("hsvImage", hsvImage)
cv2.waitKey(0)

X = np.asanyarray(temp2)
ori_pixels = X.reshape(*temp2.size, -1)
ori_pixels.shape
print(X)
