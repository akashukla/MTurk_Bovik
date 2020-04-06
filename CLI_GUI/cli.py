import sys 
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
#from Model.mid_project_demo.hw2 import *

def start_cli(cmd,args):
    print(cmd)
    print(args)
    directory = args[0]
    image_files = os.listdir(directory)
    NN = torch.load("")
    NN.predict(image_files)
    #load in neural network
    #nn.load 
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val' : transforms.Compose([
            transforms.ToTensor()
        ])
    }

    image_set = {x: datasets.ImageFolder(os.path.join(directory,""),
                    data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_set[x], 
                    #batch_size=4, shuffle=True, num_workers=4) 
                    batch_size=1, shuffle=True, num_workers=1) 
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_set[x]) for x in ['train', 'val']}
    class_names = image_set['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__": #will pass in direcory that containts images to be compressed
    print("main")
    args = sys.argv
    cmd = args[0]
    start_cli(cmd, args[1:])