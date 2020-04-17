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



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


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
import matplotlib.pyplot as plt


import itertools


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




# Compute confusion matrix


# Plot non-normalized confusion matrix

plot_confusion_matrix(cm, classes=['no oa','oa'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

