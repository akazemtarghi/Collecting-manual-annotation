
import torch.nn as nn




class Amir(nn.Module):
    def __init__(self, nclass):

        super(Amir, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(6),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=3),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU())

        self.fc = nn.Sequential(

            nn.Linear(15376, 1024),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, nclass),
            )





    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        #t = self.layer3(t)
        #t = self.layer4(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        return t

