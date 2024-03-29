# import the necessary packages
from LBP_binary_classification import lbp_calculated_pixel
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
from sklearn import svm
import os
import pandas as pd
from Utilities import SplittingData
import numpy as np
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from skimage.feature import local_binary_pattern
import scipy.stats as stats
MET = 'nri_uniform'

def plot_confusion_matrix(cm,
                          classes=['no oa', 'oa'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

def center_crop_amir(dir, new_width, new_height):

    im = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    width, height = im.shape  # Get dimensions

    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2

    cropped = im[top:bottom, left:right]


    return cropped

def upper_crop(dir, new_width, new_height):

    im = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    width, height = im.shape  # Get dimensions

    left = (width - new_width)//2
    top = 0
    right = (width + new_width)//2
    bottom = new_height

    cropped = im[top:bottom, left:right]


    return cropped
#desc = LocalBinaryPatterns(12, 2)
data = []
labels = []

from sklearn.linear_model import LogisticRegression
from Utilities import set_ultimate_seed


set_ultimate_seed(base_seed=777)
Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/med.csv'
train, test = SplittingData(Csv_dir, Ratio=0.25, all=False)
n = (len(train))
P = 18
R = 8

for i in range(n):
    temp = []
    input = train.loc[i]
    side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']

    imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/ROIs/proportional_mri/' \
                + str(label) + '/' + str(id)  + '_' + str(series) + '_' + str(side)  +'.png'
    imagePath2 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/ROIs/proportional_mri/' \
                 + str(label) + '/' + str(id)  + '_' + str(series) + '_' + 'lat'+ '.png'

    new_width = 60
    new_height = 60

    #img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE )
    # img2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)

    resized_im = upper_crop(imagePath, new_width, new_height)
    resized_im2 = upper_crop(imagePath2, new_width, new_height)
    # plt.figure()
    # plt.imshow(img)

    lbp_imag = local_binary_pattern(resized_im, P, R, method=MET)
    lbp_imag2 = local_binary_pattern(resized_im2, P, R, method=MET)

    hist1, bins = np.histogram(lbp_imag.ravel(), 256, [0, 255])
    hist2, bins = np.histogram(lbp_imag2.ravel(), 256, [0, 255])

    temp.append(hist1)
    temp.append(hist2)

    f = np.asarray(temp)



    labels.append(label)
    data.append(f.reshape(1, -1))

# train a Linear SVM on the data

aaa = np.concatenate(data, axis=0)
model = LogisticRegression(random_state=0).fit(aaa, labels)
# model = svm.SVC(kernel='rbf')
# model.fit(data, labels)


n = (len(test))
results = np.zeros((n, 2))
# loop over the testing images
for i in range(n):
    temp = []

    # load the image, convert it to grayscale, describe it,
    # and classify it
    input = test.loc[i]
    side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']

    imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/ROIs/proportional_mri/' \
                + str(label) + '/' + str(id)  + '_' + str(series) + '_' + str(side)  +'.png'
    imagePath2 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/ROIs/proportional_mri/' \
                 + str(label) + '/' + str(id)  + '_' + str(series) + '_' + 'lat'+ '.png'

    new_width = 60
    new_height = 60

    resized_im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE )
    resized_im2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)

    # resized_im = upper_crop(imagePath, new_width, new_height)
    # resized_im2 = upper_crop(imagePath2, new_width, new_height)
    # plt.figure()
    # plt.imshow(img)

    lbp_imag = local_binary_pattern(resized_im, P, R, method=MET)
    lbp_imag2 = local_binary_pattern(resized_im2, P, R, method=MET)

    hist1, bins = np.histogram(lbp_imag.ravel(), 256, [0, 255])
    hist2, bins = np.histogram(lbp_imag2.ravel(), 256, [0, 255])

    temp.append(hist1)
    temp.append(hist2)

    f = np.asarray(temp)

    #hist = desc.describe(resized_im)
    prediction = model.predict(f.reshape(1, -1))
    results[i][0] = prediction
    results[i][1] = label

from sklearn.metrics import confusion_matrix

t = results[:, 1]
p = results[:, 0]

cm = confusion_matrix(results[:, 1], results[:, 0])
cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

odds_ratio, pvalue = stats.fisher_exact(cm)

# O = np.append(O, odds_ratio)
# p = np.append(p, pvalue)
print(odds_ratio)
print(pvalue)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")


    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

plt.figure()

plot_confusion_matrix(cm, classes=['no oa', 'oa'],
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues)

plt.figure()

plot_confusion_matrix(cm, classes=['no oa', 'oa'],
                      normalize=True,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues)



