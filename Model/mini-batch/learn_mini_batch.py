from __future__ import division

# Get study data
import numpy as np
import pandas as pd

raw_set_data = pd.DataFrame.to_numpy(pd.read_csv('../../HIT-support/hit_data.csv'))
im_data = np.asarray([row[1:raw_set_data[0,0]+1] for row in raw_set_data])


# Image access helpers
from goldenlib import *

#
# Torch transfer learning
#
import os
import time
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

plt.ion()
