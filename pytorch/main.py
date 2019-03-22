import os
import torch
import torch.optim as optim
from cnn_utils import *
from Models.VGG import VGG
import torch.optim.lr_scheduler as lr_scheduler
from data_utils import *
#plot images in terminal
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from Models.resnet import *

#Use models in torchvision
#############################################################################################
#input the name of the model you want to use
#model_name = "resnet18"
#input the learning rate
learning_rate = 1e-1
# init_model(model_name, input_size, num_classes, feature_extract=True, use_pretrained=True)
# input_size is the length of the img not channel size
# if feature_extract=true, it will only finetune the changed layers
# if use_pretrained=true, it will use params of pretrained models
#model, input_size = init_model(model_name, 32, 10, feature_extract=False)
############################################################################################

#Use self-writing models
###################################
model = resnet18()
input_size = 32
#model.apply(init_weights)
###################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

dataloaders = Load_cifar10('./datasets', input_size, batch_size=128)
optimizer = optim.SGD(get_params(model), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
criterion = nn.CrossEntropyLoss()

model, val_history = train_model(
    model,
    dataloaders,
    device,
    criterion,
    optimizer,
    scheduler,
    num_epochs=300)

check_acc(model, dataloaders['test'], device)