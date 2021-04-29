# 0403.py
import cv2
import numpy as np

template = cv2.imread('../img/template.jpg', 0)
frame = cv2.imread('../img/players.jpg', 0)

cv2.imshow('Frame', frame)
cv2.imshow('Template', template)

method = cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(frame, template, method)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
print(maxVal, maxLoc)
cv2.circle(result, maxLoc, 15, 255, 2)

cv2.imshow('Matching', result)
cv2.waitKey(0)
cv2.destroyAllWindows()