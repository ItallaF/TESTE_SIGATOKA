# TESTE_SIGATOKA
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

ori_img = Image.open("C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/FolhamamaoSemFundo.jpg")
plt.imshow(ori_img)

X = np.array(ori_img.getdata())
ori_pixels = X.reshape(*ori_img.size, -1)
ori_pixels.shape
print(X)

def imageByteSize(img):
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
kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns = ['Vermelha', 'Verde', 'Azul'])
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
kmeans_df["Nome da Cor"] = list(map(get_colour_name, np.uint8(kmeans.cluster_centers_)))
print('ss', kmeans_df)

def replaceWithCentroid(kmeans):
    new_pixels = []
    for label in kmeans.labels_:
        pixel_as_centroid = list(kmeans.cluster_centers_[label])
        new_pixels.append(pixel_as_centroid)
    new_pixels = np.array(new_pixels).reshape(*ori_img.size, -1)
    return new_pixels
new_pixels = replaceWithCentroid(kmeans)
print('Novos pixelas', new_pixels)


def calculateBCSS(X, kmeans):
    _, label_counts = np.unique(kmeans.labels_, return_counts = True)
    diff_cluster_sq = np.linalg.norm(kmeans.cluster_centers_ - np.mean(X, axis = 0), axis = 1)**2
    return sum(label_counts * diff_cluster_sq)
WCSS = kmeans.inertia_
BCSS = calculateBCSS(X, kmeans)
exp_var = 100*BCSS/(WCSS + BCSS)
print("WCSS: {}".format(WCSS))
print("BCSS: {}".format(BCSS))
print("Variância explicada: {:.3f}%".format(exp_var))
print("Tamanho da imagem: {:.3f} KB".format(imageByteSize(new_pixels)))

range_k_clusters = (2, 21)
kmeans_result = []
for k in range(*range_k_clusters):
    # CLUSTERING
    kmeans = KMeans(n_clusters = k,
                    n_jobs = -1,
                    random_state = 123).fit(X)
     # REPLACE PIXELS WITH ITS CENTROID
    new_pixels = replaceWithCentroid(kmeans)
    
    # EVALUATE
    WCSS = kmeans.inertia_
    BCSS = calculateBCSS(X, kmeans)
    exp_var = 100*BCSS/(WCSS + BCSS)
    
    metric = {
        "No. de cores": k,
        "Centroids": list(map(get_colour_name, np.uint8(kmeans.cluster_centers_))),
        "Pixels": new_pixels,
        "WCSS": WCSS,
        "BCSS": BCSS,
        "Variância explicada": exp_var,
        "Tamanho da imagem (KB)": imageByteSize(new_pixels)
    }
    
    kmeans_result.append(metric)
kmeans_result = pd.DataFrame(kmeans_result).set_index("No. de cores")
kmeans_result
print('clusters', kmeans_result)

def locateOptimalElbow(x, y):
    # START AND FINAL POINTS
    p1 = (x[0], y[0])
    p2 = (x[-1], y[-1])
    
    # EQUATION OF LINE: y = mx + c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    
    # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
    a, b = m, -1
    dist = np.array([abs(a*x0+b*y0+c)/math.sqrt(a**2+b**2) for x0, y0 in zip(x,y)])
    return np.argmax(dist) + x[0]

def calculateDerivative(data):
    derivative = []
    for i in range(len(data)):
        if i == 0:
            # FORWARD DIFFERENCE
            d = data[i+1] - data[i]
        elif i == len(data) - 1:
            # BACKWARD DIFFERENCE
            d = data[i] - data[i-1]
        else:
            # CENTER DIFFERENCE
            d = (data[i+1] - data[i-1])/2
        derivative.append(d)
    return np.array(derivative)

def locateDrasticChange(x, y):
    # CALCULATE GRADIENT BY FIRST DERIVATIVE
    first_derivative = calculateDerivative(np.array(y))
    
    # CALCULATE CHANGE OF GRADIENT BY SECOND DERIVATIVE
    second_derivative = calculateDerivative(first_derivative)
    return np.argmax(np.abs(second_derivative)) + x[0]


optimal_k = []
for col in kmeans_result.columns[2:]:
    optimal_k_dict = {}
    optimal_k_dict["Métrica"] = col
    if col == "Tamanho da imagem (KB)":
        optimal_k_dict["Método"] = "Derivada"
        optimal_k_dict["K ideal"] = locateDrasticChange(kmeans_result.index, kmeans_result[col].values)
    else:
        optimal_k_dict["Método"] = "ângulo"
        optimal_k_dict["K ideal"] = locateOptimalElbow(kmeans_result.index, kmeans_result[col].values)
    optimal_k.append(optimal_k_dict)
optimal_k = pd.DataFrame(optimal_k)
optimal_k

calc = k/(ori_img_size/3)*100
print("porc", calc)
