import cv2
import numpy as np
import os
import math
import time
from matplotlib import pyplot as plt

#                Pontificia Universidad Javeriana
#               Procesamiento de imágenes y visión
#                            Taller 3
#                 Jeimmy Alejandra Cuitiva Mont
#               Laura Alejandra Estupiñan Martínez



# Si se desea guardar cada una de la imágenes generadas según el filtro,
# cambié la variable interna "path" de cada método por la dirección
# de preferencia.

class noise:
    #Constructor imagen
    def __init__(self, image_gray):
        self.image_gray = image_gray

    #Método ruido Gaussiano
    def gauss_noisy(self):
        row, col = self.image_gray.shape
        mean = 0
        var = 0.002
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        self.lena_gauss_noisy = self.image_gray + gauss
        #cv2.imshow("Gauss Noisy", self.lena_gauss_noisy)
        #Almacenar imagen con ruido resultante
        path = '/Users/lauestupinan/Desktop'
        path_file = os.path.join(path, 'Gauss_Noisy.jpg')
        img = cv2.convertScaleAbs(self.lena_gauss_noisy, alpha=(255.0))
        cv2.imwrite(path_file, img)
        #Retornar imagen
        return self.lena_gauss_noisy

    #Método ruido sANDp
    def sANDp_noisy(self):
        s_vs_p = 0.5
        amount = 0.01
        self.lena_sANDp_noisy = np.copy(self.image_gray)
        # Salt mode
        num_salt = np.ceil(amount * self.image_gray.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self.image_gray.shape]
        self.lena_sANDp_noisy[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * self.image_gray.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self.image_gray.shape]
        self.lena_sANDp_noisy[tuple(coords)] = 0
        #cv2.imshow("s&p_noisy", self.lena_sANDp_noisy)
        # Almacenar imagen con ruido resultante
        path = '/Users/lauestupinan/Desktop'
        path_file = os.path.join(path, 's&p_noisy.jpg')
        img = cv2.convertScaleAbs(self.lena_sANDp_noisy, alpha=(255.0))
        cv2.imwrite(path_file, img)
        # Retornar imagen
        return self.lena_sANDp_noisy

    #Método filtro Gaussiano
    def Gaussiano(self):
        N = 7
        sigma = 1.5

        #Filtro Gaussiano con ruido gaussiano:
        J = (255 * self.lena_gauss_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start1 = time.time()
        self.image_gauss_lp1 = cv2.GaussianBlur(J, (N, N), sigma, sigma)
        end1 = time.time()
        self.time1_1 = end1 - start1
        #print("Tiempo ejecución filtro Gaussiano con ruido gaussiano: " + str(self.time1_1))

        #Filtro Gaussiano con ruido s&p:
        L = (255 * self.lena_sANDp_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start2 = time.time()
        self.image_gauss_lp2 = cv2.GaussianBlur(L, (N, N), sigma, sigma)
        end2 = time.time()
        #cv2.imshow("Filtro Gaussiano: Gauss noisy", self.image_gauss_lp1)
        #cv2.imshow("Filtro Gaussiano: s&p noisy", self.image_gauss_lp2)
        self.time1_2 = end2 - start2
        #print("Tiempo ejecución filtro Gaussiano con ruido s&p: " + str(self.time1_2))

        #Almacenar imagen con ruido resultante
        path = '/Users/lauestupinan/Desktop'
        path_file1 = os.path.join(path, 'FG_Gauss_Noisy.jpg')
        #img1 = cv2.convertScaleAbs(self.image_gauss_lp1, alpha=(255.0))
        cv2.imwrite(path_file1, self.image_gauss_lp1)
        path_file2 = os.path.join(path, 'FG_s&p_noisy.jpg')
        #img2 = cv2.convertScaleAbs(self.image_gauss_lp2, alpha=(255.0))
        cv2.imwrite(path_file2, self.image_gauss_lp2)

        #Estimar ruido en cada caso
        self.gauss_noisy = abs(J - self.image_gauss_lp1)
        self.sANDp_noisy = abs(J - self.image_gauss_lp2)

        #Almacenar imagen estimación de ruido
        path_file3 = os.path.join(path, 'FG_Gauss_Noisy_diff.jpg')
        #img3 = cv2.convertScaleAbs(self.gauss_noisy, alpha=(255.0))
        cv2.imwrite(path_file3, self.gauss_noisy)
        path_file4 = os.path.join(path, 'FG_s&p_noisy_diff.jpg')
        #img4 = cv2.convertScaleAbs(self.sANDp_noisy, alpha=(255.0))
        cv2.imwrite(path_file4, self.sANDp_noisy)

        #Retornar imagenes junto con estimaciones de tiempo
        return self.image_gauss_lp1, self.image_gauss_lp2, self.time1_1, self.time1_2

    #Método filtro Mediana
    def Mediana(self):
        N = 7

        #Filtro Mediana con ruido gaussiano:
        J = (255 * self.lena_gauss_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start3 = time.time()
        self.image_median1 = cv2.medianBlur(J, N)
        end3 = time.time()
        self.time2_1 = end3 - start3
        #print("Tiempo ejecución filtro Mediana con ruido gaussiano: " + str(self.time2_1))

        #Filtro Mediana con ruido s&p:
        L = (255 * self.lena_sANDp_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start4 = time.time()
        self.image_median2 = cv2.medianBlur(L, N)
        end4 = time.time()
        self.time2_2 = end4 - start4
        #print("Tiempo ejecución filtro Mediana con ruido s&p: " + str(self.time2_2))
        #cv2.imshow("Filtro Mediana: Gauss noisy", self.image_median1)
        #cv2.imshow("Filtro Mediana: s&p noisy", self.image_median2)

        #Almacenar imagen con ruido resultante
        path = '/Users/lauestupinan/Desktop'
        path_file1 = os.path.join(path, 'FM_Gauss_Noisy.jpg')
        #img1 = cv2.convertScaleAbs(self.image_median1, alpha=(0))
        cv2.imwrite(path_file1, self.image_median1)
        path_file2 = os.path.join(path, 'FM_s&p_noisy.jpg')
        #img2 = cv2.convertScaleAbs(self.image_median2, alpha=(0))
        cv2.imwrite(path_file2, self.image_median2)

        #Estimar ruido en cada caso
        self.gauss_noisy = abs(J-self.image_median1)
        self.sANDp_noisy = abs(L-self.image_median2)

        #Almacenar imagen estimación de ruido
        path_file3 = os.path.join(path, 'FM_Gauss_Noisy_diff.jpg')
        cv2.imwrite(path_file3, self.gauss_noisy)
        path_file4 = os.path.join(path, 'FM_s&p_noisy_diff.jpg')
        cv2.imwrite(path_file4, self.sANDp_noisy)

        #Retornar imagenes junto con estimaciones de tiempo
        return self.image_median1, self.image_median2, self.time2_1, self.time2_2

    #Método filtro Bilateral
    def Bilateral(self):
        d = 15
        sigmaColor = 25
        sigmaSpace = 25

        #Filtro Bilateral con ruido gaussiano:
        J = (255 * self.lena_gauss_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start5 = time.time()
        self.image_bilateral1 = cv2.bilateralFilter(J, d, sigmaColor, sigmaSpace)
        end5 = time.time()
        self.time3_1 = end5 - start5

        #Filtro Bilateral con ruido s&p:
        L = (255 * self.lena_sANDp_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start6 = time.time()
        self.image_bilateral2 = cv2.bilateralFilter(L, d, sigmaColor, sigmaSpace)
        end6 = time.time()
        self.time3_2 = end6 - start6
        #cv2.imshow("Filtro Bilateral: Gauss noisy", self.image_bilateral1)
        #cv2.imshow("Filtro Bilateral: s&p noisy", self.image_bilateral2)

        #Almacenar imagen con ruido resultante
        path = '/Users/lauestupinan/Desktop'
        path_file1 = os.path.join(path, 'FB_Gauss_Noisy.jpg')
        #img1 = cv2.convertScaleAbs(self.image_bilateral1, alpha=(255.0))
        cv2.imwrite(path_file1, self.image_bilateral1)
        path_file2 = os.path.join(path, 'FB_s&p_noisy.jpg')
        #img2 = cv2.convertScaleAbs(self.image_bilateral2, alpha=(255.0))
        cv2.imwrite(path_file2, self.image_bilateral2)

        #Estimar ruido en cada caso
        self.gauss_noisy = abs(J-self.image_bilateral1)
        self.sANDp_noisy = abs(L-self.image_bilateral2)

        #Almacenar imagen estimación de ruido
        path_file3 = os.path.join(path, 'FB_Gauss_Noisy_diff.jpg')
        cv2.imwrite(path_file3, self.gauss_noisy)
        path_file4 = os.path.join(path, 'FB_s&p_noisy_diff.jpg')
        cv2.imwrite(path_file4, self.sANDp_noisy)

        #Retornar imagenes junto con estimaciones de tiempo
        return self.image_bilateral1, self.image_bilateral2, self.time3_1, self.time3_2

    #Método filtro nlm
    def nlm(self):
        h = 5
        windowSize = 15
        searchSize = 25

        #Filtro nlm con ruido gaussiano:
        J = (255 * self.lena_gauss_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start7 = time.time()
        self.image_nlm1 = cv2.fastNlMeansDenoising(J, h, windowSize, searchSize)
        end7 = time.time()
        self.time4_1 = end7 - start7

        #Filtro nlm con ruido s&p:
        L = (255 * self.lena_sANDp_noisy).astype(np.uint8)
        #Medición tiempo de ejecución
        start8 = time.time()
        self.image_nlm2 = cv2.fastNlMeansDenoising(L, h, windowSize, searchSize)
        end8 = time.time()
        self.time4_2 = end8 - start8
        #cv2.imshow("Filtro nlm: Gauss noisy", self.image_nlm1)
        #cv2.imshow("Filtro nlm: s&p noisy", self.image_nlm2)

        #Almacenar imagen con ruido resultante
        path = '/Users/lauestupinan/Desktop'
        path_file1 = os.path.join(path, 'Fnlm_Gauss_Noisy.jpg')
        #img1 = cv2.convertScaleAbs(self.image_nlm1, alpha=(255.0))
        cv2.imwrite(path_file1, self.image_nlm1)
        path_file2 = os.path.join(path, 'Fnlm_s&p_noisy.jpg')
        #img2 = cv2.convertScaleAbs(self.image_nlm2, alpha=(255.0))
        cv2.imwrite(path_file2, self.image_nlm2)
        #cv2.waitKey(0)

        #Estimar ruido en cada caso
        self.gauss_noisy = abs(J-self.image_nlm1)
        self.sANDp_noisy = abs(L-self.image_nlm2)

        #Almacenar imagen estimación de ruido
        path_file3 = os.path.join(path, 'Fnlm_Gauss_Noisy_diff.jpg')
        cv2.imwrite(path_file3, self.gauss_noisy)
        path_file4 = os.path.join(path, 'Fnlm_s&p_noisy_diff.jpg')
        cv2.imwrite(path_file4, self.sANDp_noisy)

        #Retornar imagenes junto con estimaciones de tiempo
        return self.image_nlm1, self.image_nlm2, self.time4_1, self.time4_2

    #Método cálculo raíz error cuadrático medio
    def ECM(self, I1, I2):
        M, N= I1.shape
        K = 0
        for i in range (1,M):
            for j in range (1,N):
                K+=np.power(abs(I1[i, j] - I2[i, j]),2)
        N = ((1/(M*N))*K)
        V = math.sqrt(N)
        return V

if __name__ == '__main__':

    path = '/Users/lauestupinan/Desktop'
    image_name = 'lena.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    #Convertir imagen a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Llamar constructor
    lena = noise(image_gray.astype(np.float) / 255)
    #Generar tipos de ruido
    lena_gauss_noisy = lena.gauss_noisy()
    lena_sANDp_noisy = lena.sANDp_noisy()
    #Asignar retorno métodos
    image_gauss_lp1, image_gauss_lp2, t_11, t_12 = lena.Gaussiano()
    image_median1, image_median2, t_21, t_22 = lena.Mediana()
    image_bilateral1, image_bilateral2, t_31, t_32 = lena.Bilateral()
    image_nlm1, image_nlm2, t_41, t_42 = lena.nlm()

    #Calcular raíz cuadráda error cuadráctico para filtros con ruido Gaussiano
    ECM_gauss = [0, 0, 0, 0]
    ECM_gauss[0] = lena.ECM(lena_gauss_noisy, image_gauss_lp1)
    ECM_gauss[1] = lena.ECM(lena_gauss_noisy, image_median1)
    ECM_gauss[2] = lena.ECM(lena_gauss_noisy, image_bilateral1)
    ECM_gauss[3] = lena.ECM(lena_gauss_noisy, image_nlm1)

    #print(ECM_gauss)

    #Definir filtro con ruido gaussiano de menor sqrt(ECM)
    GaussMIN= min(ECM_gauss)
    if ECM_gauss[0] == GaussMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_gauss_noisy es el Gaussiano: " + str(GaussMIN))
    elif ECM_gauss[1] == GaussMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_gauss_noisy es el Mediana: " + str(GaussMIN))
    elif ECM_gauss[2] == GaussMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_gauss_noisy es el Bilateral: " + str(GaussMIN))
    elif ECM_gauss[3] == GaussMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_gauss_noisy es el nlm: " + str(GaussMIN))

    #Calcular raíz cuadráda error cuadráctico para filtros con ruido s&p
    ECM_sANDp = [0, 0, 0, 0]
    ECM_sANDp[0] = lena.ECM(lena_sANDp_noisy, image_gauss_lp2)
    ECM_sANDp[1] = lena.ECM(lena_sANDp_noisy, image_median2)
    ECM_sANDp[2] = lena.ECM(lena_sANDp_noisy, image_bilateral2)
    ECM_sANDp[3] = lena.ECM(lena_sANDp_noisy, image_nlm2)

    #print(ECM_sANDp)

    #Definir filtro con ruido s&d de menor sqrt(ECM)
    sANDpMIN = min(ECM_sANDp)
    if ECM_sANDp[0] == sANDpMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_s&p_noisy es el Gaussiano: " + str(sANDpMIN))
    elif ECM_sANDp[1] == sANDpMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_s&p_noisy es el Mediana: " + str(sANDpMIN))
    elif ECM_sANDp[2] == sANDpMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_s&p_noisy es el Bilateral: " + str(sANDpMIN))
    elif ECM_sANDp[3] == sANDpMIN:
        print("El filtro que presenta el menor sqrt(ECM) ante la imagen lena_s&p_noisy es el nlm: " + str(sANDpMIN))

    #Generar arreglo con tiempos de ejecución en ms
    Time = [t_11*1000, t_12*1000, t_21*1000, t_22*1000, t_31*1000, t_32*1000, t_41*1000, t_42*1000]
    #Calcular menor tiempo
    TimeMIN = min(Time)

    #print(Time)

    #Encontrar menor tiempo de ejecución
    #Reportar el tiempo de ejecución del método más rápido en (ms)
    #Indicar tiempos restantes como un porcentaje referente al menor
    if Time[0] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro Gaussiano con ruido gaussiano")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[1] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro Gaussiano con ruido s&p")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0)+"%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[2] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro Mediano con ruido gaussiano")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0) + "%")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[3] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro Mediano con ruido s&p")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0) + "%")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[4] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro Bilateral con ruido gaussiano")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0) + "%")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[5] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro Bilateral con ruido s&p")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0) + "%")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[6] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro nlm con ruido gaussiano")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0) + "%")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P7 = int((Time[7] * 100) / TimeMIN)
        print("Nml con ruido s&p: " + str(P7) + "%")
    elif Time[7] == TimeMIN:
        print("El menor tiempo de ejecución fue: " + str(TimeMIN) + " correspondiente al filtro nlm con ruido s&p")
        P0 = int((Time[0] * 100) / TimeMIN)
        print("Gaussiano con ruido gaussiano: " + str(P0) + "%")
        P1 = int((Time[1] * 100) / TimeMIN)
        print("Gaussiano con ruido s&p: " + str(P1) + "%")
        P2 = int((Time[2] * 100) / TimeMIN)
        print("Mediano con ruido gaussiano: " + str(P2) + "%")
        P3 = int((Time[3] * 100) / TimeMIN)
        print("Mediano con ruido s&p: " + str(P3) + "%")
        P4 = int((Time[4] * 100) / TimeMIN)
        print("Bilateral con ruido gaussiano: " + str(P4) + "%")
        P5 = int((Time[5] * 100) / TimeMIN)
        print("Bilateral con ruido s&p: " + str(P5) + "%")
        P6 = int((Time[6] * 100) / TimeMIN)
        print("Nml con ruido gaussiano: " + str(P6) + "%")