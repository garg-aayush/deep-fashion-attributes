import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class MultiHeadResNet18(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet18, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layers according to the number of categories
        self.l0 = nn.Linear(512, 8) # for neck
        self.l1 = nn.Linear(512, 5) # for sleeve_length
        self.l2 = nn.Linear(512, 11) # for pattern
    #
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        #print(x.shape)
        x = self.model.features(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        #print(x.shape)
        l0 = self.l0(x)
        #print(l0.shape)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2
    
class MultiHeadResNet34(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet34, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layers according to the number of categories
        self.l0 = nn.Linear(512, 8) # for neck
        self.l1 = nn.Linear(512, 5) # for sleeve_length
        self.l2 = nn.Linear(512, 11) # for pattern
    #
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        #print(x.shape)
        x = self.model.features(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        #print(x.shape)
        l0 = self.l0(x)
        #print(l0.shape)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2