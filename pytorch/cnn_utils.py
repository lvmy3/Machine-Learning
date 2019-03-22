import os
import torch
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tensor
import torchvision.models as models
import torch.optim as optim


def set_parameter_requires_grad(model, feature_extracting):
    '''
    if the feature_extracting is true, set the paramters.requires_grad as False
    or else do nothing and it will finetune all the params in the model
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def init_model(model_name,
               input_dim,
               num_classes,
               feature_extract=True,
               use_pretrained=True):
    '''
    init the model

    Parameters:
    - model_name: Name of the model to use
    - input_dim: the dimension of the input
    - num_classes: the num of classes of the dataset
    - feature_extract: tune all params or just new layers
    - use_pretrained: whether to use pretrained params

    Returns:
    - model: the model initialized
    - input_size: the size used in transformation when augment data
    '''
    model = None
    input_size = input_dim

    model = getattr(models, model_name)(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)

    if "resnet" in model_name:
        #the input dim of the final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet" or "vgg" in model_name:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        init_weights(model.classifier[6])

    elif "squeezenet" in model_name:
        model.classifier[1] = nn.conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes

    elif "densenet" in model_name:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size


def train_model(model,
                dataloaders,
                device,
                criterion,
                optimizer,
                scheduler,
                num_epochs=25,
                is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += labels.size(0)

            epoch_loss = running_loss / running_total
            epoch_acc = 100 * running_corrects.double() / running_total

            print("%s loss: %.4f, Got %d / %d correct, accuracy is: %.2f%%" % (
                phase, epoch_loss, running_corrects, running_total, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                #if len(val_acc_history) > 10 and abs(epoch_acc - val_acc_history[-10]) < 5:
                #    scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def get_params(model, feature_extract=True):
    '''
    get the params that need to be tuned while training
    when feature_extract=True(default), it will only tune the params in the modified layers
    '''
    params_to_update = model.parameters()
    if feature_extract:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    return params_to_update


def init_weights(m):
    '''
    init weights for the modified layers
    '''
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def check_acc(model, loader, device):
    '''
    this can be used when check the accuracy of test set
    '''
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')

    num_total = 0
    correct_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, pred = scores.max(1)
            correct_num += (pred == y).sum()
            num_total += y.size(0)
        acc = float(correct_num) / num_total
        print(
            'Got %d / %d correct (%.2f)' % (correct_num, num_total, 100 * acc))


'''
def train_model_ver1(model, optimizer, device, criterion, dataloaders, num_epoch=25, lr=0.1):
    model = model.to(device)
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epoch):
        print("Epoch %d/%d" % (epoch + 1, num_epoch))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
                num_correct = 0
                for x, y in dataloaders[phase]:
                    x = x.to(device, dtype=torch.float32)
                    y = y.to(device, dtype=torch.int64)
                    optimizer.zero_grad()
                    scores = model(x)
                    _, pred = scores.max(1)
                    num_correct += (pred == y.data).sum()
                    loss = criterion(scores, y)
                    loss.backward()
                    optimizer.step()
                acc = float(num_correct) / len(dataloaders[phase].dataset)
                print("%s loss: %.4f, Got %d / %d correct, accuracy is: %.2f%%" % (phase,loss.item(),num_correct, len(dataloaders[phase].dataset), 100*acc))
            else:
                acc = check_acc(model, dataloaders[phase],device)
                if acc > best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(acc)
        print()
    
    time_elapsed = time.time() - since
    print("Training complete in %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: %4f" % (best_acc,))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def old_train_model(model, optimizer, device, dataloaders, epoche=25, lr=0.1):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    model = model.to(device=device)
    for epch in range(epoche):
        adjust_learning_rate(optimizer, epch, lr)
        print("Epoch %d/%d" % (epch + 1, epoche))
        print('-' * 10)
        num_correct = 0
        for x, y in dataloaders['train']:
            #send data to device
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.int64)
            #begin training:get loss, update weights
            model.train()
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            _, pred = scores.max(1)
            num_correct += (pred == y.data).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = num_correct.double() / len(dataloaders['train'].dataset)
        print('Train loss = %.4f, acc = %.2f' % (loss.item(), 100*acc))
        check_acc(model, dataloaders['val'], device)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
