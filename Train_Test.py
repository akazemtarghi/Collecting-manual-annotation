import torch
import torch.nn as nn
import numpy as np
from Utilities import EarlyStopping
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Training_dataset(data_loaders, model, patience, n_epochs,
                     namefold, tb, scheduler, optimizer, criterion):

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
