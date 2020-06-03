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
from Utilities import SplittingData,filling_dataframe_all

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

mode = ['dissimilarity', 'correlation', 'contrast', 'homogeneity', 'ASM', 'energy']
Angle = [0,  np.pi/2, np.pi/4, (3/4)*np.pi]
Distances = list(range(1, 4))

pv = []
odds_r = []
m = []
s = []
idx = 0
df = pd.DataFrame(columns=['Parameter', 'No_OA', 'No_OA_SD',
                           'OA', 'OA_SD', 'pvalue', 'ODDS_ratio'])


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


for q in mode:

    print(q)

    set_ultimate_seed(base_seed=777)
    Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/data/IDnKL.csv'
    data = pd.read_csv(Csv_dir)
    data = data.reset_index(drop=True)


    n = (len(train_set))
    feature = []
    Feature_T = []
    labels = []
    temp = []

    for i in range(n):

        input = train_set.loc[i]

        img_path = 'C:/Users/Amir Kazemtarghi/Documents/data/p26crops/' + str(input['ID']) + '.npy'

        patches, p_id = np.load(img_path)

        if input['SIDE'] == 1:
            image = patches['R']
        else:
            image = patches['L']

        windowsize_r = 29
        windowsize_c = 29
        # Crop out the window and calculate the histogram
        temp = []

        Sk = GLCM_amir(image, Angle, Distances, metric=q)

        Sk = np.asarray(Sk)
        Sk = Sk.mean()

        if input['KL'] > 1:

            input['KL'] = 1
        else:

            input['KL'] = 0



        labels.append(input['KL'])
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




    n = (len(test_set))
    results = np.zeros((n, 2))

    for i in range(n):

        input = test_set.loc[i]

        img_path = 'C:/Users/Amir Kazemtarghi/Documents/data/p26crops/' + str(input['ID']) + '.npy'

        patches, p_id = np.load(img_path)

        if input['SIDE'] == 1:
            image = patches['R']
        else:
            image = patches['L']


        Sk = GLCM_amir(image, Angle, Distances, metric=q)

        Sk = np.asarray(Sk)
        Sk = Sk.mean()

        if input['KL'] > 1:

            input['KL'] = 1
        else:

            input['KL'] = 0

        prediction = model.predict(Sk.reshape(-1, 1))
        results[i][0] = prediction
        results[i][1] = input['KL']


    from sklearn.metrics import confusion_matrix

    t = results[:, 1]
    p = results[:, 0]

    cm = confusion_matrix(results[:, 1], results[:, 0])
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    odds_ratio, pvalue = stats.fisher_exact(cm)

    odds_r.append(odds_ratio)
    pv.append(pvalue)

    df = df.append({'Parameter': q,
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
    plt.savefig('C:/Users/Amir Kazemtarghi/Desktop/New folder/'+'_CM_' + str(idx) + '.jpg', bbox_inches='tight')

    plt.figure()

    plot_confusion_matrix(cm, classes=['no oa', 'oa'],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)
    plt.savefig('C:/Users/Amir Kazemtarghi/Desktop/New folder/' + '_N_' + str(idx) + '.jpg', bbox_inches='tight')
    idx = idx + 1



df.to_csv('C:/Users/Amir Kazemtarghi/Desktop/df_GLCM.csv')