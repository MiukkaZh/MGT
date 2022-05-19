import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('model')
sys.path.append('loss')
from resnet import Linear_fw
from Log import log

class ClsLoss(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(ClsLoss, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = Linear_fw(embedding_size, num_classes)
        log.log('Initialized Classification Loss')

    def forward(self, inputs, label):
        logit = self.fc(inputs)
        loss = self.criterion(logit, label)
        acc = self.accuracy(logit, label)

        return loss, acc

    def accuracy(self, logit, label):
        answer = (torch.max(logit, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = logit.size(0)

        return answer / n_total
