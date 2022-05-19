import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('model')
sys.path.append('loss')
from resnet import Linear_fw
from Log import log

class MetaLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, use_GC=True):
        super(MetaLoss, self).__init__()

        self.zero = torch.tensor(0).cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        if use_GC:
            self.fc = Linear_fw(embedding_size, num_classes)
        self.use_GC = use_GC
        log.log('Initialized Meta Loss')

    def forward(self, support, query, label_g, label_e, model, use_GC=True, mode='normal'):

        if mode == 'normal':
            support = model(support)  # out size:(batch size, #classes), for softmax
            query = model(query)
        else:
            support = model(support, mode)  # out size:(batch size, #classes), for softmax
            query = model(query, mode)

        logit_e = F.linear(query, F.normalize(support))
        loss_e = self.criterion(logit_e, label_e)
        acc_e = self.accuracy(logit_e, label_e)

        loss_g = self.zero
        acc_g = self.zero
        if self.use_GC:
            inputs = torch.cat((support, query), dim=0)
            logit_g = self.fc(inputs)

            loss_g = self.criterion(logit_g, label_g)
            acc_g = self.accuracy(logit_g, label_g)

        loss = loss_e + loss_g

        return loss, loss_e, loss_g, acc_e, acc_g

    def accuracy(self, logit, label):
        answer = (torch.max(logit, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = logit.size(0)

        return answer / n_total
