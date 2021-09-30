# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:33:39 2021

@author: Italla

Imagem da propria biblioteca
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

rng = np.random.default_rng()

# Gerar dados sintéticos ruidosos
data = skimage.img_as_float(binary_blobs(length=128, seed=1))
sigma = 0.35
data += rng.normal(loc=0, scale=sigma, size=data.shape)
data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                         out_range=(-1, 1))

# O intervalo da imagem binária se estende por (-1, 1).
# Escolhemos os pixels mais quentes e mais frios como marcadores.
markers = np.zeros(data.shape, dtype=np.uint)
markers[data < -0.95] = 1
markers[data > 0.95] = 2

# Execute o algoritmo do walker aleatório
labels = random_walker(data, markers, beta=10, mode='bf')

# Resultados de plotagem
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(data, cmap='gray')
ax1.axis('off')
ax1.set_title('Noisy data')
ax2.imshow(markers, cmap='magma')
ax2.axis('off')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray')
ax3.axis('off')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()
