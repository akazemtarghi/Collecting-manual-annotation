# # import the necessary packages
# from skimage import feature
# import numpy as np
#
#
# class LocalBinaryPatterns:
#
#     def __init__(self, numPoints, radius):
#
#         # store the number of points and radius
#         self.numPoints = numPoints
#         self.radius = radius
#
#     def describe(self, image, eps=1e-7):
#
#         # compute the Local Binary Pattern representation
#         # of the image, and then use the LBP representation
#         # to build the histogram of patterns
#         lbp = feature.local_binary_pattern(image, self.numPoints,
#         self.radius, method="uniform")
#         (hist, _) = np.histogram(lbp.ravel(),
#         bins=np.arange(0, self.numPoints + 3),
#         range=(0, self.numPoints + 2))
#         # normalize the histogram
#         hist = hist.astype("float")
#         hist /= (hist.sum() + eps)
#         # return the histogram of Local Binary Patterns
#
#         return hist


import cv2
import numpy as np
from matplotlib import pyplot as plt


import cv2
import numpy as np
from matplotlib import pyplot as plt



def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4
    '''
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))  # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))  # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))  # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

