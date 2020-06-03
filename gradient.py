from GLCM_utilities import GLCM_amir, Histo_AMir, Gradient_amir, LAP_amir
import cv2
from Utilities import SplittingData
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from Utilities import set_ultimate_seed
import scipy.stats as stats
import image_slicer
import pandas as pd

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


'''dissimilarity
correlation
contrast
homogeneity
ASM
energy
'''
mode = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Npwzg']

ori = ['dx', 'dy', 'laplacian']

pv = []
odds_r = []
m = []
s = []
idx = 0
df = pd.DataFrame(columns=['Parameter','orientation', 'No_OA', 'No_OA_SD',
                           'OA', 'OA_SD', 'pvalue', 'ODDS_ratio'])

for q in mode:

    for w in ori:


        print(q)
        print(w)

        Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/bml_in_med.csv'
        # train, test = SplittingData(Csv_dir, all=False)
        data = pd.read_csv(Csv_dir)
        data = data.drop(columns=['Unnamed: 0'])
        n = (len(data))
        feature = []
        Feature_T = []
        labels = []
        temp = []

        for i in range(n):

            input = data.loc[i]
            side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']
            imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/ROIs/proportional_mri/' \
                        + str(label) + '/' + str(id) + '_' + str(series) + '_' + str(side)+ '.png'

            new_width = 30
            new_height = 30

            resized_im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            #resized_im = center_crop_amir(imagePath, new_width, new_height)

            windowsize_r = 29
            windowsize_c = 29
            # Crop out the window and calculate the histogram
            temp = []
            abs_gr = Gradient_amir(resized_im, mode=w)
            #Sk, Kr, Vr, Mn = Histo_AMir(resized_im)
            # mn = stats.skew(abs_gr)
            # nzCount = cv2.countNonZero(abs_gr)
            # Sk = nzCount/(abs_gr.size)

            if q == 'Mean':
                Sk = abs_gr.mean()

            elif q == 'Variance':
                Sk = abs_gr.var()

            elif q == 'Skewness':
                Sk, Kr, Vr, Mn = Histo_AMir(abs_gr)

            elif q == 'Kurtosis':
                Sk, Kr, Vr, Mn = Histo_AMir(abs_gr)
                Sk = Kr

            elif q == 'Npwzg':
                nzCount = cv2.countNonZero(abs_gr)
                Sk = nzCount/(abs_gr.size)












            #resized_im = center_crop_amir(imagePath, new_width, new_height)
            #feature = GLCM_amir(resized_im, Angle, Distances, metric='dissimilarity')


            Sk = np.asarray(Sk)

            labels.append(label)
            Feature_T.append(Sk)


        # model = svm.SVC(kernel='linear')
        # model.fit(Feature_T, labels)
        aaa = np.asarray(Feature_T)


        bbb = np.asarray(labels)
        INDEX_no = np.where(bbb == 0)
        INDEX_oa = np.where(bbb == 1)
        print(aaa[INDEX_no[0]].mean())
        print(aaa[INDEX_no[0]].std())

        print(aaa[INDEX_oa[0]].mean())
        print(aaa[INDEX_oa[0]].std())

        model = LogisticRegression(penalty='none', class_weight='balanced', solver='newton-cg')
        model.fit(aaa.reshape(-1, 1), bbb)




        n = (len(data))
        results = np.zeros((n, 2))

        for i in range(n):

            input = data.loc[i]
            side, label, series, id = input['side'], input['Label'], input['SeriesDescription'], input['ParticipantID']
            imagePath = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/ROIs/proportional_mri/' \
                        + str(label) + '/' + str(id) + '_' + str(series) + '_' + str(side) + '.png'
            resized_im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

            # resized_im = center_crop_amir(imagePath, new_width, new_height)
            # feature = GLCM_amir(resized_im, Angle, Distances, metric='dissimilarity')'

            abs_gr = Gradient_amir(resized_im, mode=w)

            if q == 'Mean':
                Sk = abs_gr.mean()

            elif q == 'Variance':
                Sk = abs_gr.var()

            elif q == 'Skewness':
                Sk, Kr, Vr, Mn = Histo_AMir(abs_gr)

            elif q == 'Kurtosis':
                Sk, Kr, Vr, Mn = Histo_AMir(abs_gr)
                Sk = Kr

            elif q == 'Npwzg':
                nzCount = cv2.countNonZero(abs_gr)
                Sk = nzCount/(abs_gr.size)

            Sk = np.asarray(Sk)
            prediction = model.predict(Sk.reshape(-1, 1))
            results[i][0] = prediction
            results[i][1] = label


        from sklearn.metrics import confusion_matrix

        t = results[:, 1]
        p = results[:, 0]

        cm = confusion_matrix(results[:, 1], results[:, 0])
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        odds_ratio, pvalue = stats.fisher_exact(cm)

        odds_r.append(odds_ratio)
        pv.append(pvalue)

        df = df.append({'Parameter': q,
                        'orientation': w,
                        'No_OA': aaa[INDEX_no[0]].mean(),
                        'No_OA_SD': aaa[INDEX_no[0]].std(),
                        'OA': aaa[INDEX_oa[0]].mean(),
                        'OA_SD': aaa[INDEX_oa[0]].std(),
                        'pvalue': pvalue,
                        'ODDS_ratio': odds_ratio}, ignore_index=True)

        print(odds_ratio)
        print(pvalue)

        plt.figure()

        plot_confusion_matrix(cm, classes=['no oa', 'oa'],
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues)
        plt.savefig('C:/Users/Amir Kazemtarghi/Desktop/New folder/' + '_CM_' + str(idx) + '.jpg', bbox_inches='tight')

        plt.figure()

        plot_confusion_matrix(cm, classes=['no oa', 'oa'],
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues)
        plt.savefig('C:/Users/Amir Kazemtarghi/Desktop/New folder/' + '_N_' + str(idx) + '.jpg', bbox_inches='tight')
        idx = idx + 1



df.to_csv('C:/Users/Amir Kazemtarghi/Desktop/df_gradient.csv')