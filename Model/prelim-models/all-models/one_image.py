# Std imports
import os, sys
import pandas as pd
import numpy as np

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, models



# PATH TO IMAGE GOES HERE
image_path = '../../data/8k_data/AVA__809598.jpg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

from PIL import Image
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device) 

image = image_loader(image_path)

# LOAD MODEL BY NAME HERE
model = torch.load('demo-for-shehryar', map_location=device)
label = model(image)

# CONTAINS THE COMPRESSION LEVEL
compression_level = label.detach().numpy().reshape(1,)[0]

