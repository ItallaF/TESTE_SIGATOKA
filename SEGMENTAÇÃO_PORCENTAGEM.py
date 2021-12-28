# TESTE_SIGATOKA
"""
Created on Wed Dec 15 22:29:47 2021

@author: Italla
"""

from PIL import Image
from io import BytesIO
import webcolors

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sb

original_image = cv2.imread("C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/FolhamamaoSemFundo.jpg")

#-----------------------Imagem BRG -------------------------------------

imagem = original_image
Z = np.float32(imagem.reshape((-1, 3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 4
ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2BRG = res.reshape((imagem.shape))
cv2.imshow("Aplicacao do kmeansBRG",res2BRG)

grayBRG = cv2.cvtColor(res2BRG, cv2.COLOR_BGR2GRAY)
cv2.imshow("Preto e branco",grayBRG)

#Função calcHist para calcular o hisograma da imagem
h = cv2.calcHist([grayBRG], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma P&B")
plt.xlabel("Intensidade")   
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()

ret3, otsu1BRG = cv2.threshold (grayBRG, 80,255, cv2.THRESH_BINARY + cv2.THRESH_BINARY)
cv2.imshow("binarizacao doente",otsu1BRG)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
maskedBRG = cv2.bitwise_and(image, image, mask=otsu1BRG)
cv2.imshow("masked brg doente",maskedBRG)

ret3, otsu2BRG = cv2.threshold (grayBRG, 80,255, cv2.THRESH_BINARY + cv2.THRESH_BINARY_INV)
cv2.imshow("binarizacao saldavel",otsu2BRG)
image = original_image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
maskedBRG = cv2.bitwise_and(image, image, mask=otsu2BRG)
cv2.imshow("masked brg saldavel",maskedBRG)

#-----------------------Fim Imagem BRG -------------------------------------
cv2.imshow("original_image",original_image)

#---------------------------Porcentagem-------------------------------------
NumeropixelsBranco = np.sum (otsu1BRG == 255) # extraindo apenas pixels brancos 
print("Numero de pixels branco", NumeropixelsBranco)
NumeropixelPreto = np.sum(otsu1BRG == 0)# extraindo apenas pixels preto 
print("Numero de pixels preto", NumeropixelPreto)
TotalPixels =  NumeropixelsBranco + NumeropixelPreto # Total de pixels
print("Total de Pixels", TotalPixels)
Porcentagem = (NumeropixelsBranco/TotalPixels) * 100
Porcentagem = round (Porcentagem, 4)
print("A porcentagen de doença é de", Porcentagem)
cv2.waitKey(0)
