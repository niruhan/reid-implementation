import torch
import torch.nn as nn
from torchvision import models


# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num=751):
        super(ft_net, self).__init__()
        # load the model
        model_ft = models.resnet50(pretrained=True)
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)  # define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)  # use our classifier.
        return x


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, num_bottleneck=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        droprate = 0.5
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x
