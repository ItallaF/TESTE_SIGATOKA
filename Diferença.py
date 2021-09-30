# TESTE_SIGATOKA
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:41:33 2021

@author: Italla

Mostra como resultado a diferença das imagens.
Imagens da propria bblioteca
"""

import matplotlib.pyplot as plt

from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert

# A imagem original é invertida porque o objeto deve ser branco
image = invert(data.horse())

chull = convex_hull_image(image)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].set_title('Foto original')
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_axis_off()

ax[1].set_title('Imagem transformada')
ax[1].imshow(chull, cmap=plt.cm.gray)
ax[1].set_axis_off()

plt.tight_layout()
plt.show()

chull_diff = img_as_float(chull.copy())
chull_diff[image] = 2

fig, ax = plt.subplots()
ax.imshow(chull_diff, cmap=plt.cm.gray)
ax.set_title('Diferença')
plt.show()
