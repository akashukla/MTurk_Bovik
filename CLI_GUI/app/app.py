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
#from torch import Image
#from torch import loader
#from torch import Variable

app = Flask(__name__)

@app.route('/')
def hello_world(): #check if app is running
    return 'Hello, World!'


@app.route('/page') #this returns the html page
def page():
    return render_template('upload_image.html')


@app.route('/upload',methods=['POST']) #use this to upload images to compress
def upload_image():
    try:
        nn_weights = 'temp.txt'
        #\NN = load_neural_net(nn_weights)

       
        
        print(request.files) #show files that user uploaded
        num_images = 0  #number of images in request
        predicted_images = []   #will store each image uri in this var
        for i in request.files:
            num_images += 1
            #print(request.files[i]) 
            #nn_predicted_image = pred_image(NN, request.files[i]) #predict compression rate given neural net and image
            # img_str = base64.b64encode(nn_predicted_image.read())
            # predicted_images.append(img_str.decode('utf-8'))#append to list

            with open('cashmoney.jpg','rb') as img: #these 3 lines are just temporary, will return a random image
                img_str = base64.b64encode(img.read())
                predicted_images.append(img_str.decode('utf-8'))

        #KEEP THE LINE BELOW COMMENTED
        #model_conv.load_state_dict(torch.load(os.path.join(os.getcwd(),"KonIQ_model"),map_location=torch.device('cpu')))

        print(len(predicted_images))
        return jsonify(
            num_images = num_images,
            image_strings = predicted_images
        ) 
    except Exception as e:
        print(e)
        return str(e)



#these 3 functions 
def load_neural_net(filename):
    n = torch.load(filename)
    return n
        
def pred_image(NN,image):
    image = image_loader(image)
    return image

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    #image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

if __name__ == "__main__":
    app.run(debug=True)