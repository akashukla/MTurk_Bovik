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
import torchvision
from torch.autograd import Variable
import fastai
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.imports import *
from sort_real_data import load_batch_data


image_names,image_scores,image_avgs,svals,savg=load_batch_data()
df=pd.DataFrame(np.array([image_names, image_avgs]).T, columns=['names', 'scores'])
df.scores=df.scores.astype('float')

img_path='../data/8k_data/'
data = (ImageList
 .from_df(path=img_path,df=df)
 .split_by_rand_pct()
 .label_from_df(cols=1,label_cls=FloatList)
 .transform(get_transforms(), size=224)
 .databunch(bs=16)
 .normalize(imagenet_stats))

learn = cnn_learner(data, models.resnet50, metrics=[accuracy], true_wd=False)
# learn.loss = MSELossFlat
# #learn.loss = L1LossFlat
# learn.fit(1)

class MSELossFlat(nn.MSELoss): 
#“Same as `nn.MSELoss`, but flattens input and target.”
  def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
   return super().forward(input.view(-1), target.view(-1))


class L1LossFlat(nn.L1Loss):
#“Mean Absolute Error Loss”
  def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
   return super().forward(input.view(-1), target.view(-1))




# learn.export('8k_epoch1.pkl')
learn.save('8k_epoch1')
learn.load('8k_epoch1')
#learn2=load_learner('../data/8k_data', '8k_epoch1.pkl')
