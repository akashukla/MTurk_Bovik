import scipy.io 
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

#
# Import LIVE Image Data
#
import re
from os import listdir
from os.path import isfile, join

head = '../../LIVE/'
paths = [d+'/' for d in listdir(head) if d not in ['.DS_Store', 'refnames_all.mat', 'dmos.mat', 'readme.txt']]
live_images = []
refimages = []

for path in paths:
    for f in listdir(head+path):
        if re._compile('.*?bmp',flags=0).match(f):
            live_images.append(head+path+f)
            if path=='refimgs/': refimages.append(f)
live_images = np.asarray(live_images)

#
# Import DMOS quality scores
#
dmos = sc.io.loadmat(head+'dmos.mat') 
refnames = sc.io.loadmat(head+'refnames_all.mat')
wn_ref = np.loadtxt(head+'wn/info.txt', dtype=str)
jp2k_ref = np.loadtxt(head+'jp2k/info.txt', dtype=str)
jpeg_ref = np.loadtxt(head+'jpeg/info.txt', dtype=str)
gblur_ref = np.loadtxt(head+'gblur/info.txt', dtype=str)
fastfading_ref = np.loadtxt(head+'fastfading/info.txt', dtype=str)


# 
# The Question:
# Can this learn what it means to be a high
# quality image? 
# What is the expected value in terms of image
# quality score and is this congruent to the 
# dmos scores?
#

# Prediction metric/output 1: 0, 1 (high quality or not)
# Prediction metric/output 2: Expected value score for an image
import cv2
import torch
from torchvision.transforms import *

binary_scores = [1-x for x in dmos['orgs'][0]]

# Set standard hyperparams - not tuned
num_epochs = 5
num_classes = 10
batch_size = 1
learning_rate = 0.01

# Apply transforms to data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081))])

train_loader = DataLoader(dataset=training_set, 
                          batch_size=batch_size,
                          shuffle=False)
test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,
                         shiffle=False)



# Use conv net model
from Net import *
model = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#
# TRAINING STEP
#

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))

#
# TESTING STEP
#

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
