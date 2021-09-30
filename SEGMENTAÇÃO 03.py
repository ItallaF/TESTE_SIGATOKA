# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:57:33 2021

@author: Italla
"""
#img = cv.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folhaBananeira.jpg')
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import data, img_as_float    
from skimage.segmentation import chan_vese

image = cv.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folhaBananeira.jpg')
#image = img_as_float('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folhaBananeira.jpg')
image = plt.imread('1.jpeg')
image.shape
plt.imshow(image)

cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                dt=0.5, init_level_set="checkerboard",
               extended_output=True)

fig, axes = plt.subplots(0, 0, figsize=(3, 3))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Imagem original", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
#Segmentação f'Chan-Vese - iterações {len (cv [4])}
title = f'Chan-Vese segmentation - {len(cv[4])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Conjunto de nível final", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolução da energia ao longo das iterações", fontsize=12)

fig.tight_layout()
plt.show()
