# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 07:38:26 2021

@author: Italla
"""
# image processing
from PIL import Image
from io import BytesIO
import webcolors

# data analysis
import math
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
from importlib import reload
from mpl_toolkits import mplot3d
import seaborn as sns

# modeling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import cv2

class Detecta_Doenca:
    def __init__(self,imagem):
        self.imagem = imagem

    def Convercao_Cores(self):
        self.imagem_YCrCb = cv2.cvtColor(self.imagem, cv2.COLOR_YCrCb2BGR)
        self.imagem_rgb = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2RGB)
        cv2.imshow("Imagem Original", self.imagem)
        cv2.imshow("Imagem em YCrCb", self.imagem_YCrCb)
        cv2.imshow("Imagem em RGB", self.imagem_rgb)
        cv2.waitKey(0)

    def Segmentar(self):
        img = self.imagem_rgb
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 5
        cluster = []

        for i in range(2,k+1):
            ret, label, center = cv2.kmeans(Z, i, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            self.res_YCrCb = center[label.flatten()]
            self.res2_YCrCb = self.res_YCrCb.reshape((img.shape))
            cv2.imshow("Imagem Original", self.imagem)
            cv2.imshow('Sigmentada do kmeans '+str(i-1), self.res2_YCrCb)
            cv2.waitKey(0)
            cluster.append(self.res2_YCrCb)
            cv2.imshow('res2_YCrCb', cluster[1])
        cv2.waitKey(0)
       
def Aplicando_Mascara(self):
        AplicandoMascaraNaImagem(self.res2_YCrCb,self.imagem)

class AplicandoMascaraNaImagem:
    def __init__(self,imagemBinarisada,ImagemOrigem):
        gray = cv2.cvtColor(imagemBinarisada, cv2.COLOR_BGR2GRAY)
        img_hsv_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        ret3, otsu1 = cv2.threshold(img_hsv_gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("binarização", otsu1)
        image = ImagemOrigem
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)  # "mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
        masked = cv2.bitwise_and(image, image, mask=otsu1)  # "image", image, mask=mask
        cv2.imshow("Imagem Original", ImagemOrigem)
        cv2.imshow("Imagem Mascarada", masked)
        cv2.waitKey(0)
        #Porcentagem(masked)
        
class Porcentagem:
    def imageByteSize(self, img):
        ori_img = Image.open(imagem1)
        plt.imshow(ori_img)
        img_file = BytesIO()
        image = Image.fromarray(np.uint8(img))
        image.save(img_file, 'png')
        return img_file.tell()/1024
    ori_img_size = imageByteSize(ori_img)
    print('tamanho original:', ori_img_size)
    ori_img_n_colors = len(set(ori_img.getdata()))
    print('número de cores:', ori_img_n_colors )

    ori_img_total_variance = sum(np.linalg.norm(X - np.mean(X, axis = 0), axis = 1)**2)
    print('Variação da imagem:', ori_img_total_variance)#Variação da imagem
    #_clusters número de centroides
    kmeans = KMeans(n_clusters = 2,
                n_jobs = -1,
                random_state = 123).fit(X)
    kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns = ['Vermelho', 'Verde', 'Azul'])
    print('ss', kmeans)
       
    def closest_colour(requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
            return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name
kmeans_df["Color Name"] = list(map(get_colour_name, np.uint8(kmeans.cluster_centers_)))
print('ss', kmeans_df)
        
imagem1 = cv2.imread("C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/FolhamamaoSemFundo.jpg")

img1 = Detecta_Doenca(imagem1)
img1.Convercao_Cores()
img1.Segmentar()
img1.Aplicando_Mascara()
img1.Porcentagem()
