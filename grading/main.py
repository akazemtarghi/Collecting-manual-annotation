import torch
import random
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.optim.lr_scheduler import StepLR
from Execution_Segmentation import Execution_Segmentation_amir
import cv2
from Utilities import SplittingData, GroupKFold_Amir, \
    roc_curve_function, tensorboardx, find_mean_std
from Dataset import OAIdataset
from network import Amir
from Train_Test import Testing_dataset, Training_dataset


if __name__ == "__main__":

    Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv'

    df = pd.read_csv(Csv_dir)

    import time
    experiment_id = time.strftime("%b_%d__%H_%M")
    print(experiment_id)
    #experiment_id = 'TensorboardX'

    os.makedirs(experiment_id, exist_ok=True)
    writer = SummaryWriter(experiment_id)

    # TODO: remove

    Mean, Std = find_mean_std(df)

    Transforms = transforms.Compose([transforms.CenterCrop(31),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[Mean], std=[Std])])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 10
    nclass = 2
    Epoch = 100
    learning_rate = 0.001

    model = Amir(nclass).to(device)



    train_Csv, test_Csv = SplittingData(Csv_dir)


    train_set = OAIdataset(csv_file=train_Csv,
                           transform=Transforms)

    test_set = OAIdataset(csv_file=test_Csv,
                          transform=Transforms)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             num_workers=1,
                                             pin_memory=False,
                                             shuffle=True)

    tensorboardx(train_set, writer, model)

    Groupkfold = GroupKFold_Amir(train_set, n_splits=5)


    y_score_sum = np.zeros((184, 2))
    # TODO: revert
    patience = 50
    nfold = 1

    for train_index, test_index in Groupkfold:
        namefold = 'Fold' + str(nfold)
        model = Amir(nclass).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        train_subset = torch.utils.data.Subset(train_set, train_index)
        valid_subset = torch.utils.data.Subset(train_set, test_index)

        trainloader = torch.utils.data.DataLoader(train_subset,
                                                  batch_size=batch_size,
                                                  num_workers=1,
                                                  pin_memory=False,
                                                  shuffle=True)

        validloader = torch.utils.data.DataLoader(valid_subset,
                                                  batch_size=batch_size,
                                                  num_workers=1,
                                                  shuffle=True,
                                                  pin_memory=False)

        data_loaders = {"train": trainloader, "val": validloader}
        data_lengths = {"train": len(trainloader), "val": len(validloader)}
        model, train_loss, valid_loss = Training_dataset(data_loaders, model, patience,Epoch,
                                                         namefold, tb=writer, #scheduler=scheduler,
                                                         optimizer=optimizer, criterion=criterion)
        y_score_sum, y = Testing_dataset(testloader, model, y_score_sum, tb=writer)
        nfold = nfold + 1
        # TODO: remove
        print("Exiting fold loop")
        break

    # Computing ROC
    roc_auc, fpr, tpr = roc_curve_function(y_score_sum, y)

    # plotting ROC
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'gray'])
    fig1 = plt.figure()
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    writer.add_figure('roc', fig1)

    print('finished')

