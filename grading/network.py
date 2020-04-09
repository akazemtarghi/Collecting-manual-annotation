
import torch.nn as nn




class Amir(nn.Module):
    def __init__(self, nclass):

        super(Amir, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=2),
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
            #nn.Dropout(),
            nn.Linear(384, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, nclass),
            nn.ReLU(inplace=True),
            )





    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        #t = self.layer3(t)
        #t = self.layer4(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        return t
