from GLCM_utilities import GLCM_amir
import numpy as np
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
from sklearn.linear_model import LogisticRegression
from Utilities import set_ultimate_seed


def center_crop_amir(dir, new_width, new_height):

    im = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    width, height = im.shape  # Get dimensions

    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2

    cropped = im[top:bottom, left:right]


    return cropped


Angle = [0, np.pi/3, np.pi/2, (3/4)*np.pi, np.pi, (3/2)*np.pi]
Distances = [1, 2, 3, 4, 5, 6, 7, 8]



set_ultimate_seed(base_seed=777)
Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv'
train, test = SplittingData(Csv_dir)
n = (len(train))
feature = []
Feature_T = []
labels = []

for i in range(n):

    input = train.loc[i]
    side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']
    imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches4/standard/' \
                + str(label) + '/' + str(id) +  '_' + str(series) + '_' + str(side) +'.png'

    new_width = 60
    new_height = 60

    resized_im = center_crop_amir(imagePath, new_width, new_height)




    feature = GLCM_amir(resized_im, Angle, Distances)

    my_feature = np.asarray(feature)



    labels.append(label)
    Feature_T.append(my_feature)


model = svm.SVC(kernel='rbf')
model.fit(Feature_T, labels)
#model = LogisticRegression(random_state=0).fit(Feature_T, labels)




n = (len(test))
results = np.zeros((n, 2))
feature = []
for i in range(n):

    input = test.loc[i]
    side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']
    imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches4/standard/' \
                + str(label) + '/' + str(id) + '_' + str(series) + '_' + str(side) + '.png'

    resized_im = center_crop_amir(imagePath, new_width, new_height)



    temp = GLCM_amir(resized_im, Angle, Distances, level=256)

    f = np.asarray(temp)


    prediction = model.predict(f.reshape(1, -1))
    results[i][0] = prediction
    results[i][1] = label


from sklearn.metrics import confusion_matrix

t = results[:, 1]
p = results[:, 0]

cm = confusion_matrix(results[:, 1], results[:, 0])
cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(cm)
print(cm_n)


