#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from simple_model import ft_net

h, w = 256, 128
data_dir = '/home/niruhan/Personal/paper/Market-1501-v15.09.15/pytorch'
batchsize = 2
num_epochs = 1
use_gpu = torch.cuda.is_available()

transform_train_list = [
    transforms.Resize((h, w), interpolation=3),
    transforms.Pad(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True, num_workers=8)
               for x in ['train', 'val']}

class_names = image_datasets['train'].classes

model = ft_net(len(class_names))
criterion = nn.CrossEntropyLoss()

lr = 0.05
optim_name = optim.SGD
ignored_params = list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
classifier_params = model.classifier.parameters()
optimizer = optim_name([
         {'params': base_params, 'lr': 0.1 * lr},
         {'params': classifier_params, 'lr': lr}
     ], weight_decay=5e-4, momentum=0.9, nesterov=True)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        # Iterate over data.
        for data in dataloaders[phase]:
            # get a batch of inputs
            inputs, labels = data
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < batchsize:  # skip the last batch
                continue
            # print(inputs.shape)
            # wrap them in Variable, if gpu is used, we transform the data to cuda.
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # -------- forward --------
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # -------- backward + optimize --------
            # only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
