import torch
import random
import torch.nn as nn
import os
import pandas as pd

from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc

import torchvision
import torch
from torch import nn
from torchviz import make_dot


def filling_dataframe(file, train_indices):

    ID = train_indices['ParticipantID']
    ID = ID.reset_index(drop=True)
    SIDE = train_indices['SeriesDescription']
    SIDE = SIDE.reset_index(drop=True)

    train_set = file[0:2 * (len(train_indices) - 1)].copy()

    for i in range(len(train_indices)):

         temp = file.loc[(file['ParticipantID'] == ID.loc[i]) &
                        (file['SeriesDescription'] == SIDE.loc[i])]
         temp = temp.reset_index(drop=True)

         train_set.loc[2 * i] = temp.loc[0]
         train_set.loc[2 * i + 1] = temp.loc[1]

    train_set = train_set.reset_index(drop=True)

    return train_set

def tensorboardx(train_dataset, writer, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              num_workers=0,
                                              batch_size=50,
                                              pin_memory=False)
    data = next(iter(trainloader))
    images = data['image']
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    model = model.to(device)
    images = images.to(device)

    make_dot(model(images), params=dict(model.named_parameters()))

    writer.add_graph(model, images)


def set_ultimate_seed(base_seed=777):
    import random
    random.seed(base_seed)

    try:
        import numpy as np
        np.random.seed(base_seed)
    except ModuleNotFoundError:
        print('Module `numpy` has not been found')
    try:
        import torch
        torch.manual_seed(base_seed + 1)
        torch.cuda.manual_seed_all(base_seed + 2)
    except ModuleNotFoundError:
        print('Module `torch` has not been found')

def SplittingData (root, Ratio = 0.20):

    """ This function split the data into train and test with rate of 4:1
        data from same ID remain in same group of train or test.


        '/home/common/Amir/test 1/ALL.csv'
    """

    file = pd.read_csv(root)
    file = file.reset_index(drop=True)
    file1 = file.copy()

    file1.drop_duplicates(subset=['ParticipantID', 'SeriesDescription'], inplace=True)

    file1 = file1.reset_index(drop=True)
    file2 = file1[['ParticipantID']+['SeriesDescription']]
    file2 = file2.sample(frac=1).reset_index(drop=True)
    Dataset_size = len(file2)
    split = int(np.floor(Ratio * Dataset_size))
    train_indices, test_indices = file2[split:], file2[:split]

    train_set = filling_dataframe(file, train_indices)
    test_set = filling_dataframe(file, test_indices)

    train_set = train_set.drop(columns=['index'])

    test_set = test_set.drop(columns=['index'])

    return train_set, test_set

def GroupKFold_Amir(input, n_splits):
    X = input
    y = X.landmarks_frame.Label[:]
    y = y.reset_index(drop=True)
    groups = X.landmarks_frame.ParticipantID[:]
    group_kfold = GroupKFold(n_splits)
    group_kfold.get_n_splits(X, y, groups)
    print(group_kfold)
    return group_kfold.split(X, y, groups)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            #self.save_checkpoint(val_loss, model)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss








def roc_curve_function(y_score_sum, y):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # creating boolean matrix instead of one array
    y_score_ave = y_score_sum / 5
    t1_indice = np.where(y == 0)
    #t2_indice = np.where(y == 1)
    #t3_indice = np.where(y == 2)
    #t4_indice = np.where(y == 3)
    #t5_indice = np.where(y == 4)

    Y = np.zeros((len(y), 5))
    Y[t1_indice, 0] = 1
    #Y[t2_indice, 1] = 1
    #Y[t3_indice, 2] = 1
    #Y[t4_indice, 3] = 1
    #Y[t5_indice, 4] = 1
    # drop the fist row which is ones
    # y_score = np.delete(y_score_ave, 0, axis=0)
    # Computing ROC and ROC AUC
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_score_ave[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc, fpr, tpr

