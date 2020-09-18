# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, dataset):
        super(LeNet, self).__init__()
        if dataset == 'mnist':
            self.conv1 = nn.Conv2d(1, 6, 5)
        else:
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if dataset == 'mnist':
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
        else:
            self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

