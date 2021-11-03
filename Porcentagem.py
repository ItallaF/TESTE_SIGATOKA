# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:04:58 2021

@author: Italla
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/FolhamamaoSemFundo.jpg")
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('img_hsv', img_hsv)

sensibilidade = 20 
limite_inferior = np.array ([50 - sensibilidade, 100, 60]) 
limite_maior = np.array ([50 + sensibilidade, 255, 255]) 
#create mask 
msk = cv.inRange (img_hsv, limite_inferior, limite_maior) 
cv.imshow('Resultado', msk)

def calcPercentage (self, msk): 
	 
	#retorna a porcentagem de branco em uma imagem binária 
	
	altura, largura = msk.shape [: 2] 
	num_pixels = altura * largura 
	count_white = cv.countNonZero (msk) 
	percent_white = (count_white / num_pixels) * 100 
	percent_white = round (percent_white, 2)
    print("Numero de pixels:", num_pixels)
    print("Numero de pixels brancos", count_white)
    print("Porcentagem de pixels branco", percent_white)
    return percent_white
