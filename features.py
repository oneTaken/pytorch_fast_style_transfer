# the vgg16 is borrowed from the pytorch tutorial
# the resent is used to reimplemented and the comparision experiments
from collections import namedtuple
import os
import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        self.model_path = './pretrained/vgg16.pth'
        if os.path.exists(self.model_path):
            vgg = models.vgg16(pretrained=False)
            vgg.load_state_dict(torch.load(self.model_path))
            vgg_pretrained_features = vgg.features
        else:
            vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Resnet101(torch.nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        self.model_path = './pretrained/resnet101.pth'
        if os.path.exists(self.model_path):
            resnet101_pretrained = models.resnet101(pretrained=False)
            pretrained_state_dict = torch.load(self.model_path)
            resnet101_pretrained.load_state_dict(pretrained_state_dict)
        else:
            resnet101_pretrained = models.resnet101(pretrained=True)
            torch.save(resnet101_pretrained, self.model_path)
        resnet101_pretrained.eval()
        self.pre    = torch.nn.Sequential()
        self.model  = list(resnet101_pretrained.children())
        self.slice1 = self.model[4]
        self.slice2 = self.model[5]
        self.slice3 = self.model[6]
        self.slice4 = self.model[7]

    def forward(self, x):
        o = x
        for i in range(4):
            o = self.model[i](o)

        o = self.slice1(o)
        feature1 = o
        o = self.slice2(o)
        feature2 = o
        o = self.slice3(o)
        feature3 = o
        o = self.slice4(o)
        feature4 = o

        return feature1, feature2, feature3, feature4