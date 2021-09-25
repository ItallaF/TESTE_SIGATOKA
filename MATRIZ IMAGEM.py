# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:43:28 2021

@author: Italla
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


imgRGB = cv.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/IMAGENS FORNECIDAS/DSC01030.jpg')

imgRGB = np.asarray(imgRGB, dtype=np.float32)/255
print("Dimensões da imagem ", imgRGB.shape)
plt.figure(figsize=(3, 3))
im = plt.imshow(imgRGB, aspect='auto')
plt.axis("off")
plt.show()

imgRGB[99:200, :, 0] = 0

plt.figure(figsize=(3, 3))
im = plt.imshow(imgRGB, aspect='auto')
plt.axis("off")
plt.show()

imgRGB[99:200, :, 1] = 0
plt.figure(figsize=(3, 3))
im = plt.imshow(imgRGB, aspect='auto')
plt.axis("off")
plt.show()

MTR = np.transpose(imgRGB[ :, :, 0])
MTG = np.transpose(imgRGB[ :, :, 1])
MTB = np.transpose(imgRGB[ :, :, 2])

MT = np.zeros((640, 480, 3))

MT[ :, :, 0 ], MT[ :, :, 1], MT[ :, :, 2] = MTR, MTG, MTB

plt.figure(figsize=(3, 3))
im = plt.imshow(imgRGB, aspect='auto')
plt.axis("off")
plt.show()

"""id = np.identity(640)
did = np.identity(640)
did[   0:100, :] = id[ 119:480, :].copy()
did[ 119:480, :] = id[   0:100, :].copy()

imgRGB[:, :, 0] = np.matmul (did, imgRGB[:, :, 0])
imgRGB[:, :, 1] = np.matmul (did, imgRGB[:, :, 1])
imgRGB[:, :, 2] = np.matmul (did, imgRGB[:, :, 2])

plt.figure(figsize=(3, 3))
im = plt.imshow(imgRGB, aspect='auto')
plt.axis("off")
plt.show()"""
