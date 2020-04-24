# import the necessary packages
from LBP_binary_classification import LocalBinaryPatterns
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

    im = Image.open(dir)
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    return im




#
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--training", required=True,
# 	help="path to the training images")
# ap.add_argument("-e", "--testing", required=True,
# 	help="path to the tesitng images")
# args = vars(ap.parse_args())
# # initialize the local binary patterns descriptor along with
# # the data and label lists
desc = LocalBinaryPatterns(9, 3)
data = []
labels = []

from sklearn.linear_model import LogisticRegression
from Utilities import set_ultimate_seed


set_ultimate_seed(base_seed=777)
Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv'
train, test = SplittingData(Csv_dir, Ratio=0.25)
n = (len(train))

for i in range(n):
    input = train.loc[i]
    side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']

    imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' \
                + str(label) + '/' + str(id) + '_' + str(side) + '_' + str(series)+ '.png'

    new_width = 31
    new_height = 31

    resized_im = center_crop_amir(imagePath, new_width, new_height)

    # load the image, convert it to grayscale, and describe it

    # image = cv2.imread(imagePath)

    #gray = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)

    hist = desc.describe(resized_im)

    #   extract the label from the image path, then update the
    # #label and data lists

    labels.append(label)
    data.append(hist)

# train a Linear SVM on the data


model = LogisticRegression(random_state=0).fit(data, labels)
# model = svm.SVC(kernel='rbf')
# model.fit(data, labels)


n = (len(test))
results = np.zeros((n, 2))
# loop over the testing images
for i in range(n):

    # load the image, convert it to grayscale, describe it,
    # and classify it
    input = test.loc[i]
    side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']

    imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' \
                + str(label) + '/' + str(id) + '_' + str(side) + '_' + str(series) + '.png'

    resized_im = center_crop_amir(imagePath, new_width, new_height)
    #gray = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)

    hist = desc.describe(resized_im)
    prediction = model.predict(hist.reshape(1, -1))
    results[i][0] = prediction
    results[i][1] = label

from sklearn.metrics import confusion_matrix
t = results[:, 1]
p = results[:, 0]
cm = confusion_matrix(results[:, 1], results[:, 0])
# # display the image and the prediction
# cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
# 			1.0, (0, 0, 255), 3)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
import seaborn as sn
import pandas as pd






fpr, tpr, _ = roc_curve(t, p)

roc_auc = auc(fpr, tpr)

# plotting ROC

lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'gray'])
fig1 = plt.figure()
for i, color in zip(range(1), colors):
    plt.plot(fpr, tpr, color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Compute confusion matrix


# Plot non-normalized confusion matrix

plt.figure()
plot_confusion_matrix(cm, classes=['no oa', 'oa'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

