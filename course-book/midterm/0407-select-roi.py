# 0407.py
import cv2 
import numpy as np 

# src = cv2.imread('../../img/spirit-week.jpg', cv2.IMREAD_GRAYSCALE)
src = cv2.imread('../../img/spirit-week.jpg')
roi = cv2.selectROI(src)
print('roi = ', roi)

replace = src[roi[1]:roi[1] + roi[3],
          roi[0]:roi[0] + roi[2]]
replace = cv2.cvtColor(replace, cv2.COLOR_BGR2GRAY)
ret, bin_img = cv2.threshold(replace, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

h,w = bin_img.shape
bin_img = np.reshape(bin_img, (h,w,1))

src[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2],:] = bin_img

cv2.imshow('img', src)
cv2.waitKey() 
cv2.destroyAllWindows()