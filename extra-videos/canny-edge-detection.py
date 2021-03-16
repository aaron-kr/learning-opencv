import numpy as np 
import cv2

img0 = cv2.imread('../img/tomatoes.jpg', 1)
img = cv2.resize(img0, (960, 540))

# Threshold - blurs multiple red tomatoes into one
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
res, thresh = cv2.threshold(hsv[:,:,0], 25, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Thresh', thresh)

# Canny edges
edges = cv2.Canny(img, 100, 70)
cv2.imshow('Canny', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()