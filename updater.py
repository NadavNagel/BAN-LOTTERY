# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ban import config


class BANUpdater(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.last_model = {} #None
        # self.last_last_model = None
        # self.last_last_last_model = None
        self.gen = 0

    def update(self, inputs, targets, criterion, to_prune, window, label='soft'):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if self.gen > 0 and label != 'hard':
            teacher_outputs = self.last_model[self.gen-1](inputs).detach()
            loss = 0.5 * self.kd_loss(outputs, targets, teacher_outputs, window) + 0.5 * criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        ##############
        if to_prune:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for name, p in self.model.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(tensor < 1e-6, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(device)
        ##############
        self.optimizer.step()
        return loss

    def register_last_model(self, weight, dataset):
        self.last_model[self.gen] = config.get_model(dataset)
        self.last_model[self.gen].load_state_dict(torch.load(weight))

    def kd_loss(self, outputs, labels, teacher_outputs, window, alpha=0.2, T=20):
        if window > 0 and int(self.gen) >= 2:
            if self.gen == 2 or (int(self.gen)> 2 and window == 2):
                KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                         F.softmax(((teacher_outputs[self.gen] +
                                                    teacher_outputs[self.gen-1])/2) / T, dim=1)) * \
                        alpha + F.cross_entropy(outputs, labels) * (1. - alpha)
            if int(self.gen)> 2 and window == 3:
                KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                         F.softmax(((teacher_outputs[self.gen]+
                                                    teacher_outputs[self.gen - 1]+
                                                    teacher_outputs[self.gen - 2])/3) / T, dim=1)) * \
                        alpha + F.cross_entropy(outputs, labels) * (1. - alpha)
        else:
            KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs[self.gen] / T, dim=1)) * \
                      alpha + F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss

    def __model(self):
        return self.model

    def __last_model(self):
        return self.last_model

    def __gen(self):
        return self.gen
