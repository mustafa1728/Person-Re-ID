from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

import torch.nn as nn
from torchvision.models import segmentation

class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()


    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))

class AlignedReIDModel(nn.Module):
    def __init__(self, num_classes = 62, loss={'softmax'}, aligned=False, **kwargs):
        super(AlignedReIDModel, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        # if not self.training:
        #     return f,lf
        # y = self.classifier(f)
        f = f.view([1] + list(f.size()))
        return f
        
from torchvision.models.segmentation import fcn_resnet50
segmentation_model = fcn_resnet50(pretrained=True, progress=False)

sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

class MaskAlignedReIDModel(nn.Module):
    def __init__(self, num_classes = 62, loss={'softmax'}, aligned=False, **kwargs):
        super(MaskAlignedReIDModel, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()

        self.merge_conv = nn.Conv2d(4, 3, kernel_size=7, stride=2, padding=3,bias=False)

        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        pred = segmentation_model(x)['out']
        normalized_masks = torch.nn.functional.softmax(pred, dim=1)
        person_masks = normalized_masks[:, sem_class_to_idx['person']]
        person_masks = person_masks.unsqueeze(1)
        x = torch.cat((x, person_masks), 1)
        x = self.merge_conv(x)

        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        # if not self.training:
        #     return f,lf
        # y = self.classifier(f)
        f = f.view([1] + list(f.size()))
        return f
        