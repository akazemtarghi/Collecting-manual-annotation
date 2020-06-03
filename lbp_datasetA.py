# import the necessary packages
from LBP_binary_classification import lbp_calculated_pixel
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
from sklearn import svm
import os
import pandas as pd
from Utilities import SplittingData,filling_dataframe_all
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

def filling_dataframe_all(file, train_indices):

    ID = train_indices['ID']
    ID = ID.reset_index(drop=True)

    train_set = file[0:2 * (len(train_indices) - 1)].copy()

    for i in range(len(train_indices)):

         temp = file.loc[(file['ID'] == ID.loc[i])]
         temp = temp.reset_index(drop=True)

         train_set.loc[2 * i] = temp.loc[0]
         train_set.loc[2 * i + 1] = temp.loc[1]

    train_set = train_set.drop(columns='Unnamed: 0')
    train_set = train_set.reset_index(drop=True)
    train_set['KL'][train_set['KL'] < 2] = 0
    train_set['KL'][train_set['KL'] > 1] = 1

    return train_set


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
Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/data/IDnKL.csv'
file = pd.read_csv(Csv_dir)
file = file.reset_index(drop=True)
file_copy = file.copy()
file.drop_duplicates(subset=['ID'], inplace=True)
file = file.reset_index(drop=True)
file = file[['ID']+['SIDE']]
msk = np.random.rand(len(file)) < 0.80
train_indices = file[msk]
test_indices = file[~msk]

train_set = filling_dataframe_all(file_copy, train_indices)
test_set = filling_dataframe_all(file_copy, test_indices)

n = (len(train_set))
P = 8
R = 2




for i in range(n):

    temp = []
    input = train_set.loc[i]

    img_path = 'C:/Users/Amir Kazemtarghi/Documents/data/p26crops/' + str(input['ID']) + '.npy'

    patches, p_id = np.load(img_path)

    if input['SIDE'] == 1:
        image = patches['R']
    else:
        image = patches['L']

    # resized_im = upper_crop(imagePath, new_width, new_height)
    # resized_im2 = upper_crop(imagePath2, new_width, new_height)
    # plt.figure()
    # plt.imshow(img)

    lbp_imag = local_binary_pattern(image, P, R)
    # lbp_imag2 = local_binary_pattern(resized_im2, P, R, method=MET)

    hist1, bins = np.histogram(lbp_imag.ravel(), 256, [0, 255])
    # hist2, bins = np.histogram(lbp_imag2.ravel(), 256, [0, 255])

    temp.append(hist1)
    # temp.append(hist2)

    f = np.asarray(temp)



    labels.append(input['KL'])
    data.append(f.reshape(1, -1))

# train a Linear SVM on the data

aaa = np.concatenate(data, axis=0)
model = LogisticRegression(random_state=0).fit(aaa, labels)
# model = svm.SVC(kernel='rbf')
# model.fit(data, labels)


n = (len(test_set))
results = np.zeros((n, 2))
# loop over the testing images
for i in range(n):
    temp = []

    temp = []
    input = test_set.loc[i]

    img_path = 'C:/Users/Amir Kazemtarghi/Documents/data/p26crops/' + str(input['ID']) + '.npy'

    patches, p_id = np.load(img_path)

    if input['SIDE'] == 1:
        image = patches['R']
    else:
        image = patches['L']

    # resized_im = upper_crop(imagePath, new_width, new_height)
    # resized_im2 = upper_crop(imagePath2, new_width, new_height)
    # plt.figure()
    # plt.imshow(img)

    lbp_imag = local_binary_pattern(image, P, R)
    # lbp_imag2 = local_binary_pattern(resized_im2, P, R, method=MET)

    hist1, bins = np.histogram(lbp_imag.ravel(), 256, [0, 255])
    temp.append(hist1)

    f = np.asarray(temp)

    #hist = desc.describe(resized_im)
    prediction = model.predict(f.reshape(1, -1))
    results[i][0] = prediction
    results[i][1] = input['KL']

from sklearn.metrics import confusion_matrix

t = results[:, 1]
p = results[:, 0]

cm = confusion_matrix(results[:, 1], results[:, 0])
cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

odds_ratio, pvalue = stats.fisher_exact(cm)

# O = np.append(O, odds_ratio)
# p = np.append(p, pvalue)
print(odds_ratio)
print('{:.20f}'.format(pvalue))

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


def roc_curve_function(y, y_score_sum):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # creating boolean matrix instead of one array

    t1_indice = np.where(y == 1)
    # t2_indice = np.where(y == 1)
    # t3_indice = np.where(y == 2)
    # t4_indice = np.where(y == 3)
    # t5_indice = np.where(y == 4)

    Y = np.zeros((len(y), 1))

    yy = np.zeros((len(y), 1))

    Y[:, 0] = y_score_sum

    Y[t1_indice, 0] = 1
    # Y[t2_indice, 1] = 1
    # Y[t3_indice, 2] = 1
    # Y[t4_indice, 3] = 1
    # Y[t5_indice, 4] = 1
    # drop the fist row which is ones
    #y_score = np.delete(y_score_ave, 0, axis=0)
    # Computing ROC and ROC AUC'
    Y.reshape(-1, 1)
    y_score_sum.reshape(-1, 1)
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y, y_score_sum)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc, fpr, tpr

roc_auc, fpr, tpr = roc_curve_function(t, p)

# plotting ROC
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','gray'])
fig1 = plt.figure()
for i, color in zip(range(1), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()