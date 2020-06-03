import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
import numpy as np
import cv2
from radiomics import glrlm
import scipy.stats as stats

def GLCM_amir(patch, Angle , Distances, level=256, metric=None):

    feature = []

    for i in Angle:

        for j in Distances:



            glcm = greycomatrix(patch, distances=[j],
                                angles=[i], levels=level,
                                symmetric=True, normed=True)
            feature.append(greycoprops(glcm, metric)[0, 0])

    return feature

def Histo_AMir(img):

    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    Sk = stats.skew(hist)
    Kr = stats.kurtosis(hist)
    Vr = hist.var()
    Mn = hist.mean()

    return Sk, Kr, Vr, Mn




def LAP_amir(img):


    out = cv2.Laplacian(img,cv2.CV_64F,ksize=3)



    return out

def Gradient_amir(img, mode=None):

    #out1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    #out2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    if mode == 'dx':
        sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel64f = np.absolute(sobelx64f)
        out1 = np.uint8(abs_sobel64f)
        return out1

    if mode == 'dy':

        sobelx64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel64f = np.absolute(sobelx64f)
        out2 = np.uint8(abs_sobel64f)
        return out2

    if mode == 'laplacian':

        out3 = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        return out3





def grlm():
    feature = grlm()


