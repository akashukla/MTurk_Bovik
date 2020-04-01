import sys 
import torch
import torch.nn as nn
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




if __name__ == "__main__": #will pass in direcory that containts images to be compressed
    print("main")
    args = sys.argv
    cmd = args[0]
    start_cli(cmd, args[1:])