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

# def tensorboardx(train_dataset, writer, model):
#
#     trainloader = torch.utils.data.DataLoader(train_dataset,
#                                               num_workers=0,
#                                               batch_size=50,
#                                               pin_memory=False)
#     data = next(iter(trainloader))
#     images = data['image']
#     grid = torchvision.utils.make_grid(images)
#     writer.add_image('images', grid, 0)
#     images = images.to(device)
#     writer.add_graph(model, images, verbose=False)

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

def SplittingData (root, Ratio = 0.25):

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


class OAIdataset(Dataset):
    """datasetA."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with KL grade and ID.
            root_dir (string): Directory with all the images.
        """
        self.landmarks_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        Input = self.landmarks_frame.loc[idx]

        imageID = Input['ParticipantID']
        landmarks = Input['Label']
        side = str(Input['side'])
        id = str(imageID)

        img = cv2.imread('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' +
                         str(landmarks) +'/' + '/' + id + '_' + side + '.png')


        colaboption = '/content/gdrive/My Drive/atad/standard/' + landmarks + '/' + id +'_' + side + '.png'

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'landmarks': landmarks, 'imageID': imageID}

        return sample


class Amir(nn.Module):
    def __init__(self, nclass):

        super(Amir, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, nclass),
            #nn.ReLU(inplace=True)
        )



    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        t = self.layer3(t)
        #t = self.layer4(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        return t


def Training_dataset(data_loaders, model, patience, n_epochs, namefold, tb):

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    lr_tb = []

    for epoch_idx in range(n_epochs):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        print('LR:', scheduler.get_lr())


        tb.add_scalars(namefold + "/lr",
                       {'lr': np.asarray(scheduler.get_lr())}, epoch_idx)





        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                scheduler.step(epoch=epoch_idx)
            else:
                model.train(False)  # Set model to evaluate mode
            # Iterate over data.

            data_loader = data_loaders[phase]
            for batch in data_loader:
                optimizer.zero_grad()
                # get the input images and their corresponding labels
                images = batch['image']
                key_pts = batch['landmarks']
                images = images.to(device)
                key_pts = key_pts.to(device)

                # wrap them in a torch Variable
                output_pts = model(images)

                # calculate the loss between predicted and target keypoints
                loss = criterion(output_pts, key_pts)
                # zero the parameter (weight) gradients
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    train_losses.append(loss.item())

                else:
                    # print loss statistics
                    valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        tb.add_scalars(namefold + "/loss",
                       {'train': train_loss,
                        'valid': valid_loss}, epoch_idx)

        lr_array = np.asarray(lr_tb, dtype=float)




        # TODO: implement


        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch_idx:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f}' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses


def Testing_dataset(testloader, model,y_score_sum, tb):
    y_score = torch.ones(1, 2)
    y = []
    y_score = y_score.cpu().numpy()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images = data['image']
            label = data['landmarks']
            images = images.to(device)
            label1 = label.to(device)
            output1 = model(images)
            softmax = nn.Softmax()
            output2 = softmax(output1)
            _, predicted = torch.max(output2.data, 1)
            output2 = output2.cpu().numpy()
            label2 = label1.cpu().numpy()
            y_score = np.append(y_score, output2, axis=0)
            y = np.append(y, label2, axis=0)
            total += label1.size(0)
            correct += (predicted == label1).sum().item()

        y_score = np.delete(y_score, 0, axis=0)
        y_score_sum = y_score_sum + y_score
        #print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        print('Next fold#################################')
        return y_score_sum, y


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



if __name__ == "__main__":
    import time
    experiment_id = time.strftime("%b_%d__%H_%M")
    print(experiment_id)
    #experiment_id = 'TensorboardX'

    os.makedirs(experiment_id, exist_ok=True)
    writer = SummaryWriter(experiment_id)

    # TODO: remove

    Transforms = transforms.Compose([transforms.Resize([64, 64], interpolation=2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.456],std=[0.224])])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    batch_size = 50
    nclass = 2
    Epoch = 100
    learning_rate = 0.001

    model = Amir(nclass).to(device)

    train_Csv, test_Csv = SplittingData('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv')

    train_set = OAIdataset(csv_file=train_Csv,
                           root_dir='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/DatabaseA/',
                           transform=Transforms)

    test_set = OAIdataset(csv_file=test_Csv,
                          root_dir='C:/Users/Amir Kazemtarghi/Documents/INTERNSHIP/data/DatabaseA/',
                          transform=Transforms)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             pin_memory=False)

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
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        train_subset = torch.utils.data.Subset(train_set, train_index)
        valid_subset = torch.utils.data.Subset(train_set, test_index)

        trainloader = torch.utils.data.DataLoader(train_subset,
                                                  batch_size=batch_size,
                                                  num_workers=0,
                                                  pin_memory=False)
        validloader = torch.utils.data.DataLoader(valid_subset,
                                                  batch_size=batch_size,
                                                  num_workers=0,
                                                  pin_memory=False)

        data_loaders = {"train": trainloader, "val": validloader}
        data_lengths = {"train": len(trainloader), "val": len(validloader)}
        model, train_loss, valid_loss = Training_dataset(data_loaders, model, patience, Epoch, namefold, tb=writer)
        y_score_sum, y = Testing_dataset(testloader, model, y_score_sum, tb=writer)
        nfold = nfold + 1

        # TODO: remove
        print("Exiting fold loop")
        break

    # Computing ROC
    roc_auc, fpr, tpr = roc_curve_function(y_score_sum, y)

    # plotting ROC
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','gray'])
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


    # visualize the loss as the network trained
    fig2 = plt.figure()
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # fig2.savefig('loss_plot.png', bbox_inches='tight')
    writer.add_figure('minimum loss', fig2)