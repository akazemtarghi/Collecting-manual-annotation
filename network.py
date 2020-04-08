
import torch.nn as nn




class Amir(nn.Module):
    def __init__(self, nclass):

        super(Amir, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(1024, nclass),
            nn.ReLU(inplace=True),
            )

        self.dropout1 = nn.Dropout2d(0.25)



    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        #t = self.layer3(t)
        #t = self.layer4(t)
        x = self.dropout1(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        return t

