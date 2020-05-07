import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data



def GLCM_amir(patch, Angle , Distances, level=256):

    feature = []

    for i in Angle:

        for j in Distances:



            glcm = greycomatrix(patch, distances=[j],
                                angles=[i], levels=256,
                                symmetric=True, normed=True)

            feature.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            feature.append(greycoprops(glcm, 'correlation')[0, 0])
            feature.append(greycoprops(glcm, 'contrast')[0, 0])
            feature.append(greycoprops(glcm, 'homogeneity')[0, 0])
            feature.append(greycoprops(glcm, 'ASM')[0, 0])
            feature.append(greycoprops(glcm, 'energy')[0, 0])


    return feature




