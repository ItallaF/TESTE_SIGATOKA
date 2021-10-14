# TESTE_SIGATOKA
import numpy as np
import cv2
import matplotlib.pyplot as plt

class converteImagemParaYCrCb:
    def __init__(self,imagem):
        self.imagemYCrCb = cv2.cvtColor(imagem, cv2.COLOR_YCrCb2BGR)

class converteImagemParaRGB:
    def __init__(self,imagem):
        self.imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

class converteImagemParaYIQ:
    def __init__(self,imagemRGB):
       w, h ,c= imagemRGB.shape
       print(imagemRGB.shape)
       Y = np.zeros((w, h))
       I = np.zeros((w, h))
       Q = np.zeros((w, h))
       for i in range(0, w):
           for j in range(0, h):
               R = imagemRGB[i, j, 2]
               G = imagemRGB[i, j, 1]
               B = imagemRGB[i, j, 0]
               # RGB -> YIQ
               Y[i,j] = int((0.299*R) + (0.587 * G) + (0.114 * B))
               I[i,j] = int((0.596 * R) - (0.274 * G) - (0.322 * B))
               Q[i,j] = int((0.211 * R) - (0.523 * G) + (0.312 * B))
               yiq = cv2.merge((Y, I, Q))
               self.img_out = yiq.astype(np.uint8)
 
class aplicandoKmeansNaImagem:
    def __init__(self, imagem):
        Z = np.float32(imagem.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 10
        ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        self.res2 = res.reshape((imagem.shape))

class converteImagemParaTonsDeCinza:
    def __init__(self,imagem):
        self.imagemEmCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

class aplicaSuavizacaoNaImagemComMetodoGaussiano:
    def __init__(self,imagem):
        self.imagemSuavizada = cv2.GaussianBlur(imagem, (5, 5), 0)

"""class retiraBordas:
    def __init__(self,imagemSuavizada):
        bordas =  cv2.Canny(imagemSuavizada,100,200)
        ret, thresh = cv2.threshold (bordas, 127, 255, 0)
        contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours (imagem, contornos, -1, (255,0,255), 3)
        print("Numeros de contornos = " + str(len(contornos)))
        
        cnt = contornos[0]
        img2 = cv2.drawContours(imagem, [cnt], 0, (0,255,0), 3)
        mask = np.zeros(imagem.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (50,50,450,290)
        cv2.grabCut(imagem,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img1 = img2*mask2[:,:,np.newaxis]"""
        
class aplicaBinarizacaoDaImagemMetodoOtsu:
    def __init__(self,imagem):
        imagem = converteImagemParaTonsDeCinza(imagem).imagemEmCinza
        ret3, self.imagemOtsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

class aplicaMacara:
    def __init__(self,imagemOriginal,imagemBinarizada):
        mascara = np.zeros(imagemOriginal.shape[:2], dtype="uint8")
        cv2.rectangle(mascara, (0, 90), (290, 450), 255, -1)  # "mask"= endica em qual imagem sera aplicado, (0, 90), (290, 450), 255, 5)
        self.imagemComMascara = cv2.bitwise_and(imagemOriginal, imagemOriginal, mask=imagemBinarizada)

class aplicandoFuncao: #Função não esta fucionando é preciso verificar
    def __init__(self,imagem):
        imagemEmCinza = converteImagemParaTonsDeCinza(imagem).imagemEmCinza
        print("OK")
        imagemSuavizada = aplicaSuavizacaoNaImagemComMetodoGaussiano(imagemEmCinza).imagemSuavizada
        print("OK")
        imagemBinarizada = aplicaBinarizacaoDaImagemMetodoOtsu(imagemSuavizada).imagemOtsu
        print("OK")
        self.imagemResposta = aplicaMacara(imagem,imagemBinarizada)

        #POSSÍVEL FORMULA DE CACULAR A PORCENTAGEM D DOENÇA
print("número de pixels na planta:",len(yiq.nonzero()[0]))
# distância é 50
distance_top=1000
Area=(pow((0.000122*(distance_top-0.304)/0.304),2)*len(yiq.nonzero()[0]))
print("folha area:",round(Area, 2))
print("Porcentagem da doença: ", len(yiq.nonzero()[0])*100)

img1 = cv2.imread("C:/Users/Italla/Documents/Italla/SI/PROJETOS/SIGATOKA/TESTES/img/FolhamamaoSemFundo.jpg")
a = aplicandoKmeansNaImagem(img1).res2
cv2.imshow("img1",a)
cv2.waitKey(0)
