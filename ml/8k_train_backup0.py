import cv2
import urllib
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Access golden image access methods
import pandas as pd
from goldenlib import *
import numpy.random as npr
from numpy.random import permutation, shuffle
#From arkan's hw q1
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
#From arkan's hw q2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

### Import golden image data
golden_resI = pd.read_csv('golden_results_old.csv')
golden_resII = pd.read_csv('golden_results_new.csv')

datI = np.asarray([golden_resI.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resI.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)
datII = np.asarray([golden_resII.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_resII.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)

### Calculate thresholds
# Averages and stdevs of all images
muI = np.mean(datI, axis=0)
muII = np.mean(datII, axis=0)

avg_stdevI = (1/len(datI))*np.sum(np.std(datI, axis=0))
avg_stdevII = (1/len(datII))*np.sum(np.std(datII, axis=0))

# Normalize subject ranges to expected values
diffI = [np.subtract(datI[i],muI) for i in range(len(datI))]
diffII = [np.subtract(datII[i],muII) for i in range(len(datII))]

norm_rangeI = np.max(diffI, axis=1) - np.min(diffI, axis=1)
norm_rangeII = np.max(diffII, axis=1) - np.min(diffII, axis=1)

# Use Spearman's correlation on Golden Image Study participants to test 
# ONLY using Golden Image Study 2 from here on out
import scipy as sp
golden_dat = np.delete(datII,17,axis=0) # Removing the really terrible outlier
golden_mu = np.mean(golden_dat, axis=0)
golden_std = np.mean(golden_dat, axis=0)

repeated_thresh = (1/len(golden_dat))*np.sum(np.std(golden_dat, axis=0))

ordered_means_i = np.argsort(golden_mu)
ordered_means = np.sort(golden_mu)
avg_corr = (1/len(golden_dat))*np.sum([sp.stats.spearmanr(ordered_means,golden_dat[i,:])[0] for i in range(len(golden_dat))]) 

# Select most contrasted mean values and obtain srcc_thresh
contra_mu_names=['JPEGImages__2008_006461.jpg', 'VOC2012__2009_003324.jpg','JPEGImages__2011_004079.jpg','EMOTIC__COCO_val2014_000000147471.jpg','VOC2012__2011_003123.jpg']
contra_mu_i=[np.argwhere(golden_names==contra_mu_names[i])[0,0] for i in range(len(contra_mu_names))]
contra_mu = golden_mu[contra_mu_i]
contra_std = golden_std[contra_mu_i]
avg_contra_corr = (1/len(golden_dat))*np.sum([sp.stats.spearmanr(contra_mu,golden_dat[i,contra_mu_i])[0] for i in range(len(golden_dat))]) 
srcc_thresh = avg_contra_corr

#Load 8k data
images = np.genfromtxt('8k_image_names.csv','str',delimiter=',')[:,0]

# Image Set Params
num_images = images.shape[0]            # total number of images in 8k dataset
set_size = 45                           # number of images per set
num_golden = 5                          # number of goldens per set
num_repeat = 5                          # number of repeats per set
data_points = 30                        # number of responses per image
num_total_sets = (num_images*data_points)/set_size   # total number of sets
#Remove Goldens From Main Set
ginx = [np.argwhere(images == golden_names[i])[0,0] for i in range(len(golden_names))]
images = images[np.delete(np.arange((images.shape[0])),ginx)]
#Repeat Params
rep_first_min = 0                       # Lower end of first repeated window
rep_first_max = 10                      # Upper end of first repeated window
rep_second_min = 20                     # Lower end of second repeated window
rep_second_max = 30                     # Upper end of second repeated window
num_images = images.shape[0]            # total number of images in 8k dataset

#batch = np.genfromtxt('Batch_3941286_batch_results.csv','str',delimiter=',')[:,0]
#batch_scores = np.asarray([batch.loc[:,'Answer.slider_values'][i].split(',') 
#    for i in range(len(golden_resII.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)
    
#Generate golden learning set
classes=np.arange(0,51).astype('str')

# os.mkdir('sorted_data')
# os.mkdir('sorted_data/train')
# os.mkdir('sorted_data/val')
# for class_i in classes:
#      os.mkdir('sorted_data/train/'+class_i)
#      os.mkdir('sorted_data/val/'+class_i)
#      
# for i in range(len(golden_names)):
#      dest_tr = 'sorted_data/' + '/train/'+ (golden_mu[i].round().astype('int')).astype('str')
#      dest_va = 'sorted_data/' + '/val/' + (golden_mu[i].round().astype('int')).astype('str')
#      os.system('cp 8k_data/%s %s'%(golden_names[i], dest_tr))
#      os.system('cp 8k_data/%s %s'%(golden_names[i], dest_va))
    
"""Creating neural network to train on this dataset"""
kon_path = 'sorted_data/'

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'val' : transforms.Compose([
        transforms.ToTensor()
    ])
}

image_set = {x: datasets.ImageFolder(os.path.join(kon_path,x),
                data_transforms[x])
                for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_set[x], 
                #batch_size=4, shuffle=True, num_workers=4) 
                batch_size=1, shuffle=True, num_workers=1) 
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_set[x]) for x in ['train', 'val']}
class_names = image_set['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#plt.ion()   # interactive mode

print(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    class_dict=(dataloaders['train'].dataset.class_to_idx)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                labels = torch.tensor([class_dict['%s'%labels.item()]]).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    print(outputs)
                    print(labels)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    class_dict=(dataloaders['val'].dataset.class_to_idx)
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                #ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                ax.set_title('predicted: %s, actual: %d'%(class_names[preds[j]], labels))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 50)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=3)


## #train_model(model, criterion, optimizer, scheduler, num_epochs=25):
## model=model_conv
## optimizer=optimizer_conv
## num_epochs=25
## since = time.time()
## 
## best_model_wts = copy.deepcopy(model.state_dict())
## best_acc = 0.0
## phase='train'
## class_dict=(dataloaders[phase].dataset.class_to_idx)
## for epoch in range(num_epochs):
##     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
##     print('-' * 10)
## 
##     # Each epoch has a training and validation phase
##     for phase in ['train', 'val']:
##         if phase == 'train':
##             model.train()  # Set model to training mode
##         else:
##             model.eval()   # Set model to evaluate mode
## 
##         running_loss = 0.0
##         running_corrects = 0
## 
##         # Iterate over data.
##         for inputs, labels in dataloaders[phase]:
##         #for i in len(dataloaders[phase]):
##             inputs = inputs.to(device)
##             #labels = torch.tensor([class_dict['%s'%labels.item()]]).to(device)
##             labels = labels.to(device)
## 
##             # zero the parameter gradients
##             optimizer.zero_grad()
## 
##             # forward
##             # track history if only in train
##             with torch.set_grad_enabled(phase == 'train'):
##                 outputs = model(inputs)
##                 _, preds = torch.max(outputs, 1)
##                 print(outputs)
##                 print(labels)
##                 loss = criterion(outputs, labels)
## 
##                 # backward + optimize only if in training phase
##                 if phase == 'train':
##                     loss.backward()
##                     optimizer.step()
## 
##             # statistics
##             running_loss += loss.item() * inputs.size(0)
##             running_corrects += torch.sum(preds == labels.data)
##         if phase == 'train':
##             scheduler.step()
## 
##         epoch_loss = running_loss / dataset_sizes[phase]
##         epoch_acc = running_corrects.double() / dataset_sizes[phase]
## 
##         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
##             phase, epoch_loss, epoch_acc))
## 
##         # deep copy the model
##         if phase == 'val' and epoch_acc > best_acc:
##             best_acc = epoch_acc
##             best_model_wts = copy.deepcopy(model.state_dict())
## 
##     print()
## 
## time_elapsed = time.time() - since
## print('Training complete in {:.0f}m {:.0f}s'.format(
##     time_elapsed // 60, time_elapsed % 60))
## print('Best val Acc: {:4f}'.format(best_acc))
## 
## # load best model weights
## model.load_state_dict(best_model_wts)


visualize_model(model_conv)

plt.show()
