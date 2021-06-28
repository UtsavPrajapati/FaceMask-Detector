# -*- coding: utf-8 -*-
"""
Created on Wed May 12 20:09:58 2021

@author: user
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os
import urllib.request as urllib2
import ssl

URL = "https://192.168.1.107:8080/shot.jpg"

model = load_model("dataset2.hdf5")


imagefile="dataset"
trainsub="mask"


index=220

colors = [(0,255,0),(0,0,255)]
classes = ["mask","maskless"]
def write(name,image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    global index
    Name = name+"-"+str(index)+".png"
    
    while Name in os.listdir(imagefile):
        Name = name+"-"+str(index)+".png"
        
        index += 1
    cv2.imwrite(imagefile+os.path.sep+Name,gray)
    print(index)
        
    
def resize(img):
 
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()



def check_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = image_to_feature_vector(gray) / 255.0
    features = np.array([features])

    probs = model.predict(features)[0]
    
    prediction = probs.argmax(axis=0)
    color = colors[prediction]
    text = classes[prediction]
    return color,text



def detect_face(frame, block=False):
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    area = 0
    X = Y = W = H = 0
    roi=frame
    for (x, y, w, h) in faces:
        if w * h > area:
            area = w * h
            y=y-50
            h = h+70
            X, Y, W, H = x, y, w, h
            roi = frame[y:(y+h),x:(x+w)]
            try:
                color,text = check_mask(roi)
                cv2.rectangle(frame, (X, Y), (X + W, Y + H), color, 3)
                cv2.putText(frame,text , (X, Y-20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 3)
            except:
                pass
    return frame,roi

cap = cv2.VideoCapture(0)

while True:
    req = urllib2.Request(URL)
    gcontext = ssl.SSLContext()
    
    
    img_arr = np.array(bytearray(urllib2.urlopen(URL, context=gcontext).read()),dtype=np.uint8)
   
    
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    
    if frame is not None:
        #remove that annoying ulto effect
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = resize(frame)
        frame,roi = (detect_face(frame))
        cv2.imshow("frame",frame)
   
        wk = cv2.waitKey(1)
    
        if wk == ord('a'):
            cap.release()
            cv2.destroyAllWindows()
            break
        elif wk == ord('s'):
            im = roi
            write(trainsub,im)