import torch
import random
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.optim.lr_scheduler import StepLR
from Utilities import find_mean_std
from Utilities import SplittingData, GroupKFold_Amir, \
    roc_curve_function, tensorboardx, set_ultimate_seed
from Dataset import OAIdataset
from network import Amir
from Train_Test import Testing_dataset, Training_dataset
from Stratified_Group_k_Fold import stratified_group_k_fold, get_distribution

if __name__ == "__main__":

    Csv_dir = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv'

    df = pd.read_csv(Csv_dir)

    set_ultimate_seed(base_seed=777)

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
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(7),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[Mean], std=[Std])])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 50
    nclass = 2
    Epoch = 1000
    learning_rate = 0.001

    model = Amir(nclass).to(device)



    train_Csv, test_Csv = SplittingData(Csv_dir)


    train_set = OAIdataset(csv_file=train_Csv,
                           transform=Transforms)

    test_set = OAIdataset(csv_file=test_Csv,
                          transform=Transforms)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             pin_memory=False,
                                             shuffle=True)


    tensorboardx(train_set, writer, model)

    Notransforms = transforms.Compose([transforms.CenterCrop(31),
                                       transforms.ToTensor()])

    just_for_display = OAIdataset(csv_file=train_Csv,
                           transform=Notransforms)

    tensorboardx(just_for_display, writer, model)

    Groupkfold = GroupKFold_Amir(train_set, n_splits=5)


    y_score_sum = np.zeros((110, 2))
    # TODO: revert
    patience = 50
    nfold = 1

    train_x = train_set
    train_y = train_x.landmarks_frame.Label[:]
    train_y = train_y.reset_index(drop=True)
    groups = train_x.landmarks_frame.ParticipantID[:]


    distrs = [get_distribution(train_y)]
    index = ['training set']

    for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(train_x, train_y, groups, k=5)):

        dev_y, val_y = train_y[dev_ind], train_y[val_ind]
        dev_groups, val_groups = groups[dev_ind], groups[val_ind]

        assert len(set(dev_groups) & set(val_groups)) == 0

        distrs.append(get_distribution(dev_y))
        index.append(f'development set - fold {fold_ind}')
        distrs.append(get_distribution(val_y))
        index.append(f'validation set - fold {fold_ind}')



        namefold = 'Fold' + str(fold_ind)
        model = Amir(nclass).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        train_subset = torch.utils.data.Subset(train_set, dev_ind)
        valid_subset = torch.utils.data.Subset(train_set, val_ind)

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
        model, train_loss, valid_loss = Training_dataset(data_loaders, model, patience, Epoch,
                                                         namefold, tb=writer,  # scheduler=scheduler,
                                                         optimizer=optimizer, criterion=criterion)
        y_score_sum, y = Testing_dataset(testloader, model, y_score_sum, tb=writer)
        nfold = nfold + 1
        # TODO: remove
        print("Exiting fold loop")
        break

    print('Distribution per class:')
    print(pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(train_y) + 1)]))



    # Computing ROC
    roc_auc, fpr, tpr = roc_curve_function(y_score_sum, y)

    # plotting ROC
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'gray'])
    fig1 = plt.figure()
    for i, color in zip(range(1), colors):
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
    #writer.add_figure('roc', fig1)

    print('finished')

