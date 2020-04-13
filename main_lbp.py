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
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []


Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv'
train, test = SplittingData(Csv_dir, Ratio=0.25)
n = (len(train))

for i in range(n):

	input = train.loc[i]
	side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']

	imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' \
				+ str(label) + '/' + str(id) + '_' + str(side) + '_' + str(series)+'.png'


	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	labels.append(label)
	data.append(hist)

# train a Linear SVM on the data
model = svm.SVC(kernel='rbf')
model.fit(data, labels)


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


	image = cv2.imread(imagePath, cv2.COLOR_BGR2GRAY)
	dim = (64, 64)
	resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

	hist = desc.describe(resized)
	prediction = model.predict(hist.reshape(1, -1))
	results[i][0] = prediction
	results[i][1] = label



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(results[:][1], results[:][0])
	# # display the image and the prediction
	# cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	# 			1.0, (0, 0, 255), 3)
	# cv2.imshow("Image", image)
	# cv2.waitKey(0)

import seaborn as sn
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(cm, index = [i for i in [['No OA'], ['OA']]],
                  columns = [i for i in [['No OA'], ['OA']]])

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)