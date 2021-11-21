import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary


# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num=751):
        super(ft_net, self).__init__()
        # load the model
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        classifier = nn.Sequential(*[nn.Linear(2048, class_num)])
        self.classifier = classifier

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

print(summary(ft_net(), input_size=(2, 3, 256, 128)))