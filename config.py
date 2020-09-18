# -*- coding: utf-8 -*-
import os
import torch
from ban.models.mlp import MLP
from ban.models.lenet import LeNet
from ban.models.resnet import ResNet18
from ban.models.resnet import ResNet34
from ban.models.resnet import ResNet50
from ban.models.resnet import ResNet101
from ban.models.resnet import ResNet152

"""
add your model.
from your_model_file import Model
model = Model()
"""

# model = ResNet50()
def get_model(dataset, path=None, gen=None, epoch=None):
    model = LeNet(dataset)
    if gen is not None:
        weight_path = os.path.join(path,"gen_" + str(gen) ,dataset + "_gen_" + str(gen) + "_epoch_" + str(epoch) + ".pth.tar")
        model.load_state_dict(torch.load(weight_path))
    return model