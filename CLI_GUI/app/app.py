from flask import Flask
from flask import render_template
from flask import request
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world(): #check if app is running
    return 'Hello, World!'


@app.route('/page') #this returns the html page
def page():
    return render_template('upload_image.html')

@app.route('/upload',methods=['POST']) #use this to upload images to compress
def upload_image():
    # image = request.data
    # nparr = np.fromstring(image,np.uint8)
    # return str(nparr)
    image = request.files
    print(image)
    return image



if __name__ == "__main__":
    app.run(debug=True)