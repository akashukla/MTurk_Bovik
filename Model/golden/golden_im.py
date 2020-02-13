import cv2
import os, sys
import urllib.request
import numpy as np


def get_im(imlink):
    try: 
        resp = urllib.request.urlopen(base_url + imlink)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image
  
    # Catching the exception generated     
    except Exception as e : 
        print(str(e)) 

base_url = 'http://snako.s3.us-east-2.amazonaws.com/' 

golden_names = np.asarray(['EMOTIC__COCO_train2014_000000208055.jpg',
                         'EMOTIC__COCO_train2014_000000211272.jpg',
                         'EMOTIC__COCO_train2014_000000211486.jpg', 
                         'EMOTIC__COCO_train2014_000000212083.jpg',
                         'EMOTIC__COCO_train2014_000000221881.jpg', 
                         'EMOTIC__COCO_train2014_000000222140.jpg',
                         'EMOTIC__COCO_train2014_000000223888.jpg', 
                         'EMOTIC__COCO_train2014_000000226597.jpg',
                         'EMOTIC__COCO_train2014_000000229188.jpg', 
                         'EMOTIC__COCO_train2014_000000230433.jpg',
                         'EMOTIC__COCO_train2014_000000239307.jpg',
                         'EMOTIC__COCO_train2014_000000241421.jpg',
                         'EMOTIC__COCO_train2014_000000250357.jpg',
                         'EMOTIC__COCO_train2014_000000257297.jpg',
                         'EMOTIC__COCO_train2014_000000265100.jpg',
                         'EMOTIC__COCO_train2014_000000266479.jpg',
                         'EMOTIC__COCO_train2014_000000268209.jpg',
                         'EMOTIC__COCO_train2014_000000276128.jpg',
                         'EMOTIC__COCO_train2014_000000277122.jpg',
                         'EMOTIC__COCO_train2014_000000297359.jpg',
                         'EMOTIC__COCO_train2014_000000300598.jpg',
                         'EMOTIC__COCO_train2014_000000301855.jpg',
                         'EMOTIC__COCO_train2014_000000302102.jpg',
                         'EMOTIC__COCO_train2014_000000304548.jpg',
                         'EMOTIC__COCO_train2014_000000305105.jpg',
                         'EMOTIC__COCO_train2014_000000307894.jpg',
                         'EMOTIC__COCO_train2014_000000308353.jpg',
                         'EMOTIC__COCO_train2014_000000311706.jpg',
                         'EMOTIC__COCO_train2014_000000318496.jpg',
                         'EMOTIC__COCO_train2014_000000319690.jpg',
                         'EMOTIC__COCO_train2014_000000319905.jpg',
                         'EMOTIC__COCO_train2014_000000322212.jpg',
                         'EMOTIC__COCO_train2014_000000325981.jpg',
                         'EMOTIC__COCO_train2014_000000326504.jpg',
                         'EMOTIC__COCO_train2014_000000327810.jpg',
                         'EMOTIC__COCO_train2014_000000329587.jpg',
                         'EMOTIC__COCO_train2014_000000329942.jpg',
                         'EMOTIC__COCO_train2014_000000334338.jpg',
                         'EMOTIC__COCO_train2014_000000341623.jpg',
                         'EMOTIC__COCO_train2014_000000341905.jpg',
                         'EMOTIC__COCO_train2014_000000342969.jpg',
                         'EMOTIC__COCO_train2014_000000344031.jpg',
                         'EMOTIC__COCO_train2014_000000347133.jpg',
                         'EMOTIC__COCO_train2014_000000349698.jpg',
                         'EMOTIC__COCO_train2014_000000351610.jpg',
                         'EMOTIC__COCO_train2014_000000353483.jpg',
                         'EMOTIC__COCO_train2014_000000355425.jpg',
                         'EMOTIC__COCO_train2014_000000360017.jpg',
                         'EMOTIC__COCO_train2014_000000361190.jpg',
                         'EMOTIC__COCO_train2014_000000362658.jpg',
                         'EMOTIC__bj6qnim3c43cj2j7n3.jpg',
                         'EMOTIC__COCO_train2014_000000051735.jpg',
                         'VOC2012__2007_000063.jpg',
                         'VOC2012__2007_000068.jpg',
                         'VOC2012__2007_000793.jpg',
                         'VOC2012__2007_001487.jpg',
                         'VOC2012__2007_001583.jpg',
                         'VOC2012__2007_001704.jpg',
                         'VOC2012__2007_002565.jpg',
                         'VOC2012__2007_002643.jpg',
                         'VOC2012__2007_003104.jpg',
                         'VOC2012__2008_005594.jpg',
                         'VOC2012__2008_005600.jpg',
                         'VOC2012__2008_005607.jpg',
                         'VOC2012__2008_005615.jpg'])
