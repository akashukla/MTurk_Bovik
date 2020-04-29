from flask import Flask
from flask import render_template
from flask import request
from flask import send_file
import base64
import mimetypes
import numpy as np
from flask import jsonify
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import PIL 
from torch import tensor
#import '../../Model/prelim-models/all-models/demo-for-shehryar'
# from torch import Image
# from torch import loader
#from torchvision import Variable
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, models
import io
import json


app = Flask(__name__)


# @app.route('/')
# def hello_world(): #check if app is running
#     return 'Hello, World!'


@app.route('/') #this returns the html page
def page():
    return render_template('upload_image.html')


@app.route('/upload',methods=['POST']) #use this to upload images to compress
def get_compression_level():
    try:
        nn_weights = '../../Model/prelim-models/all-models/demo-for-shehryar'
        device = get_device()
        
        model = torch.load('../../Model/prelim-models/all-models/demo-for-shehryar', map_location=device)
        
        print(request.files) #show files that user uploaded
        num_images = 0  #number of images in request
        predicted_images = []   #will store each image uri in this var
        compression_levels = []
        for i in request.files:
            file_storage_obj = request.files[i]
            # with open(file_storage_obj,'r'):
            #     print(file_storage_obj.read())
            num_images += 1
            img_str = io.BytesIO(file_storage_obj.read())#base64.b64encode(file_storage_obj.read())
            print('i: ',img_str) 

            img = load_image(img_str,device)
            label = model(img)
            print(label)
            compression_level = label.detach().numpy().reshape(1,)[0]
            print(compression_level)
            #nn_predicted_image = pred_image(NN, request.files[i],device) #predict compression rate given neural net and image
            compression_levels.append(compression_level)
            #print(nn_predicted_image)
            #img_str = base64.b64encode(nn_predicted_image.read())
            #predicted_images.append(img_str.decode('utf-8'))#append to list

            # with open('cashmoney.jpg','rb') as img: #these 3 lines are just temporary, will return a random image
            #     img_str = base64.b64encode(img.read())
            #     predicted_images.append(img_str.decode('utf-8'))

        #KEEP THE LINE BELOW COMMENTED
        #model_conv.load_state_dict(torch.load(os.path.join(os.getcwd(),"KonIQ_model"),map_location=torch.device('cpu')))

        print(compression_levels)
        return json.dumps(str(compression_levels))
        # return jsonify(
        #     compression_levels = str(compression_levels)
        # ) 
    except Exception as e:
        print(e)
        return str(e)



#these 3 functions 
def get_device():
    #torch.nn.Module.dump_patches=True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = torch.load(filename, map_location=device)
    
    return device
        
def load_image(img,device):
    imsize = 256
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    image = Image.open(img)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device) 

# def pred_image(NN,image,device):
#     image = image_loader(image,device)
#     label = model(image)
#     compression_level = label.detach().numpy().reshape(1,)[0]

#     return compression_level

# def image_loader(image_name,device):
#     try:
#         """load image, returns cuda tensor"""

#         imsize =256
#         loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
        
#         image = Image.open(image_name)
#         print(image)
#         image = Variable(image, requires_grad=True)
#         image = image.unsqueeze(0)
#         return image.to(device) 
#         #image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        
#         #return image.cpu()#image.cuda()  #assumes that you're using GPU
#     except Exception as e:
#         print(e)

# nn_weights = '../../Model/prelim-models/all-models/demo-for-shehryar'
# NN,device = load_neural_net(nn_weights)

if __name__ == "__main__":
    app.run(debug=True)
