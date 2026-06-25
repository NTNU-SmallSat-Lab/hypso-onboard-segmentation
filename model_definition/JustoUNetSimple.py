import torch
import torch.nn as nn
import torch.nn.functional as F


class JustoUNetSimple(nn.Module):

    def __init__(self, in_ch, num_classes):
        super().__init__()
        
        # encoder
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # upsampling blocks
        self.up1   = nn.Upsample(scale_factor=2, mode="nearest")  # Keras UpSampling2D
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(6)

        self.up2   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = nn.Conv2d(6, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(num_classes)

        self.relu  = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.up1(x)
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.up2(x)
        x = self.bn4(self.conv4(x))  # logits (no softmax here)
        return x
