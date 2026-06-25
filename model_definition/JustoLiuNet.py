import torch
import torch.nn as nn
import torch.nn.functional as F


class JustoLiuNet(nn.Module):
    def __init__(self, NUMBER_OF_FEATURES, NUMBER_OF_CLASSES, KERNEL_SIZE=6):
        super().__init__()

        k = 6

        self.conv1 = nn.Conv1d(1, k, kernel_size=KERNEL_SIZE)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(k, k*2, kernel_size=KERNEL_SIZE)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(k*2, k*3, kernel_size=KERNEL_SIZE)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(k*3, k*4, kernel_size=KERNEL_SIZE)
        self.pool4 = nn.MaxPool1d(2)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, NUMBER_OF_FEATURES)
            dummy = self._forward_features(dummy)
            self.flatten_dim = dummy.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, NUMBER_OF_CLASSES)

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self._forward_features(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x