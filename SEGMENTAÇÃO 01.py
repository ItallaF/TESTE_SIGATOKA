# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:09:42 2021

@author: Italla
"""

"""#plt.imshow(img)

#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#plt.imshow(img)
#img = io.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folhaBananeira.jpg')
plt.imshow(img, cmap='gray')
#io.imsave('logo.png', logo) Salva a imagem
img = img.open(r"IMG_PATH")
img = img.filter(ImageFilter.GaussianBlur)
img.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    
    Converte uma imagem RGB em espaço de cor YIQ
    : param imgRGB: Uma imagem em RGB
    : return: Um YIQ no espaço de cores da imagem
    
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape=imgRGB.shape
    return np.dot(imgRGB.reshape(-1,3), yiq_from_rgb.transpose()).reshape(OrigShape)

    pass

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    
    Converte uma imagem YIQ em espaço de cor RGB
    : param imgYIQ: Uma imagem no YIQ
    : return: Um RGB no espaço de cores da imagem
    
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape=imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1,3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)

    pass
"""

#from skimage import data 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from PIL import Image, ImageFilter
import colorsys

imgRGB = cv.imread('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/folhaBananeira.jpg')

imgRGB = cv.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
#p1r = imgRGB[:, :, 0]
#p1g = imgRGB[:, :, 1]
#p31b = imgRGB[:, :, 1]

#img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#img_yiq = colorsys.rgb_to_yiq(0, 1, 1)
#plt.imshow ((colorsys.rgb_to_yiq * 255))
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
yiq_from_rgb = yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                                        [0.59590059, -0.27455667, -0.32134392],
                                        [0.21153661, -0.52273617, 0.31119955]])

YIQ = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb).reshape(imgRGB.shape)

return YIQ

y1y = (0.299*p1r + 0.587*p1g + 0.114*p1b)
y1i = (0.596*p1r - 0.275*p1g - 0.321*p1b)
y1q = (0.212*p1r - 0.523*p1g + 0.311*p1b)

cv.imshow("y1y", y1y)
cv.imshow("y1i", y1i)
cv.imshow("y1q", y1q)
