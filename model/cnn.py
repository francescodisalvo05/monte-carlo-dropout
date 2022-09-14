import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # layer 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # layer 3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # layer 4
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # dropout
        self.dropout1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64*13*13, 2)
        self.sig1 = nn.Sigmoid()

    def forward(self, x):

        # layer 1
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

        # layer 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

        # layer 3
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # layer 4
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))

        # flatten
        x = x.view(32, -1)

        x = self.sig1(self.fc1(x))

        return x