import cv2
import os, sys
import urllib
import urllib.request
import numpy as np


base_url = 'http://snako.s3.us-east-2.amazonaws.com/' 
def get_im(imlink):
    resp = urllib.request.urlopen(base_url + imlink)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

#cv2.imshow('first', get_im(golden_names[0]))
#cv2.waitKey(0)


golden_names = np.array(['AVA__20278.jpg', 'EMOTIC__4wl5mafxcb0nacmnzr.jpg', 'EMOTIC__COCO_train2014_000000571644.jpg', 'EMOTIC__COCO_val2014_000000031983.jpg', 'EMOTIC__COCO_val2014_000000045685.jpg', 'EMOTIC__COCO_val2014_000000147471.jpg', 'EMOTIC__COCO_val2014_000000464134.jpg', 'EMOTIC__COCO_val2014_000000534041.jpg', 'JPEGImages__2007_003137.jpg', 'JPEGImages__2008_002845.jpg', 'JPEGImages__2008_003688.jpg', 'JPEGImages__2008_003701.jpg', 'JPEGImages__2008_004077.jpg', 'JPEGImages__2008_005266.jpg', 'JPEGImages__2008_006076.jpg', 'JPEGImages__2008_006461.jpg', 'JPEGImages__2008_006554.jpg', 'JPEGImages__2008_006946.jpg', 'JPEGImages__2008_007218.jpg', 'JPEGImages__2008_008544.jpg', 'JPEGImages__2009_000015.jpg', 'JPEGImages__2009_001078.jpg', 'JPEGImages__2009_002872.jpg', 'JPEGImages__2009_002893.jpg', 'JPEGImages__2009_004426.jpg', 'JPEGImages__2010_000088.jpg', 'JPEGImages__2010_002618.jpg', 'JPEGImages__2010_003648.jpg', 'JPEGImages__2010_004257.jpg', 'JPEGImages__2011_000180.jpg', 'JPEGImages__2011_002511.jpg', 'JPEGImages__2011_004079.jpg', 'JPEGImages__2011_006006.jpg', 'JPEGImages__2011_007016.jpg', 'JPEGImages__2011_007090.jpg', 'JPEGImages__2012_000805.jpg', 'JPEGImages__2012_003188.jpg', 'JPEGImages__2012_003896.jpg', 'JPEGImages__2012_004289.jpg', 'VOC2012__2008_001175.jpg', 'VOC2012__2008_001855.jpg', 'VOC2012__2008_002589.jpg', 'VOC2012__2008_003036.jpg', 'VOC2012__2008_004536.jpg', 'VOC2012__2008_008339.jpg', 'VOC2012__2008_008503.jpg', 'VOC2012__2009_000151.jpg', 'VOC2012__2009_000659.jpg', 'VOC2012__2009_000802.jpg', 'VOC2012__2009_001550.jpg', 'VOC2012__2009_002175.jpg', 'VOC2012__2009_002794.jpg', 'VOC2012__2009_003177.jpg', 'VOC2012__2009_003324.jpg', 'VOC2012__2009_004324.jpg', 'VOC2012__2010_000853.jpg', 'VOC2012__2010_005622.jpg', 'VOC2012__2011_002173.jpg', 'VOC2012__2011_002215.jpg', 'VOC2012__2011_002960.jpg', 'VOC2012__2011_003123.jpg', 'VOC2012__2011_003969.jpg', 'VOC2012__2011_005123.jpg', 'VOC2012__2012_001367.jpg', 'VOC2012__2012_003683.jpg'])

golden_names_old = np.asarray(['EMOTIC__COCO_train2014_000000208055.jpg',
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
