import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np


EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
with open('new_fer.json.txt', 'r') as json_file:
    json_savedModel= json_file.read()
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('fer.h5')

cap = cv2.VideoCapture(0)


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        face_image = cv2.resize(roi_gray,(48,48))
        pred = np.argmax(model_j.predict(face_image.reshape(1,48,48,1))[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, EMOTIONS[pred], (x,y-50),  font, 1,(255,255,255),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(300) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()