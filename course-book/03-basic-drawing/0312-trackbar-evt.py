import cv2 
import numpy as np 

def onChange(pos): # trackbar handler 
  global img 
  r = cv2.getTrackbarPos('R', 'img')
  g = cv2.getTrackbarPos('G', 'img')
  b = cv2.getTrackbarPos('B', 'img')
  img[:] = (b,g,r)
  cv2.imshow('img', img)

img = np.zeros((512, 512, 3), np.uint8)
cv2.imshow('img', img)

# Create trackbars
cv2.createTrackbar('R', 'img', 0, 255, onChange)
cv2.createTrackbar('G', 'img', 0, 255, onChange)
cv2.createTrackbar('B', 'img', 0, 255, onChange)

# Reset trackbar position
cv2.setTrackbarPos('R', 'img', 0)
cv2.setTrackbarPos('G', 'img', 0)
cv2.setTrackbarPos('B', 'img', 255)

cv2.waitKey()
cv2.destroyAllWindows()