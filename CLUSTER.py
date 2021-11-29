# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:41:14 2021

@author: Italla
"""

from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster

NUM_CLUSTERS = 5

print('reading image')
im = Image.open('C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/ImagemYCrCbCortada.png')
im = im.resize((150, 150))      # opcional, para reduzir o tempo
ar = np.asarray(im)
shape = ar.shape
ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

print('encontrando clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
print('centros de brilho:\n', codes)

vecs, dist = scipy.cluster.vq.vq(ar, codes)         # atribuir códigos
counts, bins = scipy.histogram(vecs, len(codes))    # contagem de ocorrências

index_max = scipy.argmax(counts)                    # encontrar mais frequente
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
print('mais frequente é %s (#%s)' % (peak, colour))