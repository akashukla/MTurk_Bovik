#
# The purpose of this script is to 
# make it ammenable to visually 
# inspect the data from the 
# golden image study.
#
# Author: MK Swaminathan
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import golden image data
golden_res = pd.read_csv('./golden_results.csv')

dat = np.asarray([golden_res.loc[:,'Answer.slider_values'][i].split(',') 
    for i in range(len(golden_res.loc[:,'Answer.slider_values']))])[:,:-1].astype(int)

