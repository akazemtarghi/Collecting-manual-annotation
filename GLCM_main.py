from GLCM_utilities import GLCM_amir, Histo_AMir
import cv2
from Utilities import SplittingData
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from Utilities import set_ultimate_seed
import scipy.stats as stats
import image_slicer



def center_crop_amir(dir, new_width, new_height):

    im = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

    width, height = im.shape  # Get dimensions

    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2

    cropped = im[top:bottom, left:right]


    return cropped


'''dissimilarity
correlation
contrast
homogeneity
ASM
energy
'''

Angle = [0,  np.pi/2, np.pi/4, (3/4)*np.pi, np.pi]
Distances1 = list(range(1, 31))
windowsize_r = 4
windowsize_c = 4

p = []
O = []

for a in range(len(Distances1)):
    Distances = Distances1[0:a+1].copy()

    Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/med.csv'
    train, test = SplittingData(Csv_dir, all=False)

    n = (len(train))
    feature = []
    Feature_T = []
    labels = []
    temp = []

    for i in range(n):

        input = train.loc[i]
        side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']
        imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' \
                    + str(label) + '/' + str(id) + '_' + str(side) + '_' + str(series) +  '.png'

        new_width = 30
        new_height = 30

        #resized_im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

        windowsize_r = 15
        windowsize_c = 15

        # Crop out the window and calculate the histogram
        # temp = []
        # for r in range(0, resized_im.shape[0] - windowsize_r, windowsize_r):
        #     for c in range(0, resized_im.shape[1] - windowsize_c, windowsize_c):
        #         window = resized_im[r:r + windowsize_r, c:c + windowsize_c]
        #         Sk, Kr, Vr, Mn = Histo_AMir(window)
        #         my_feature = np.asarray(Vr)
        #         temp.append(my_feature)



        resized_im = center_crop_amir(imagePath, new_width, new_height)

        feature = GLCM_amir(resized_im, Angle, Distances, metric='contrast')


        my_f = np.asarray(feature)
        labels.append(label)
        Feature_T.append(my_f.reshape(1, -1))


    # model = svm.SVC(kernel='linear')
    # model.fit(Feature_T, labels)
    aaa = np.concatenate(Feature_T, axis=0)
    model = LogisticRegression(random_state=0).fit(aaa, labels)




    n = (len(test))
    results = np.zeros((n, 2))

    for i in range(n):

        input = test.loc[i]
        side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']
        imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' \
                    + str(label) + '/' + str(id) + '_' + str(side) + '_' + str(series) +  '.png'

        #resized_im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

        resized_im = center_crop_amir(imagePath, new_width, new_height)



        feature = GLCM_amir(resized_im, Angle, Distances, metric='contrast')
        # temp = []
        # for r in range(0, resized_im.shape[0] - windowsize_r, windowsize_r):
        #     for c in range(0, resized_im.shape[1] - windowsize_c, windowsize_c):
        #         window = resized_im[r:r + windowsize_r, c:c + windowsize_c]
        #         Sk, Kr, Vr, Mn = Histo_AMir(window)
        #         my_feature = np.asarray(Vr)
        #         temp.append(my_feature)

        f = np.asarray(feature)


        prediction = model.predict(f.reshape(1, -1))
        results[i][0] = prediction
        results[i][1] = label


    from sklearn.metrics import confusion_matrix

    t = results[:, 1]
    p = results[:, 0]

    cm = confusion_matrix(results[:, 1], results[:, 0])
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    odds_ratio, pvalue = stats.fisher_exact(cm)

    O = np.append(O, odds_ratio)
    p = np.append(p, pvalue)
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

    # plt.figure()
    #
    # plot_confusion_matrix(cm, classes=['no oa', 'oa'],
    #                       normalize=False,
    #                       title='Confusion matrix',
    #                       cmap=plt.cm.Blues)
    #
    # plt.figure()
    #
    # plot_confusion_matrix(cm, classes=['no oa', 'oa'],
    #                       normalize=True,
    #                       title='Confusion matrix',
    #                       cmap=plt.cm.Blues)
    #
    #
    #
