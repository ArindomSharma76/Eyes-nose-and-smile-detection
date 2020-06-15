import numpy as np
import cv2
import  imutils

face_cascade = cv2.CascadeClassifier('C:/Users/Arindom/Desktop/New folder (2)/har/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Arindom/Desktop/New folder (2)/har/haarcascade_eye.xml')

img = cv2.imread('C:/Users/Arindom/Desktop/New folder (2)/face.jpg')

# Resize the image - change width to 500
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.CascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbors)
faces = face_cascade.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),
                      (0,0,255),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()