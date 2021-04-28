# 0406.py
import cv2
import numpy as np

img = cv2.imread('../img/faces.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path = 'haarcascade_eye.xml'

eye_casc = cv2.CascadeClassifier(path)

eyes = eye_casc.detectMultiScale(gray, scaleFactor = 1.02, minNeighbors = 20, minSize = (10,10))
print(len(eyes))

for (x,y,w,h) in eyes:
    cx = (2*x+w) / 2
    cy = (2*y+h) / 2
    r = w/2
    cv2.circle(img, (int(cx),int(cy)), int(r), (0,255,0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()