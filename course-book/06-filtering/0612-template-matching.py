# 0612.py
import cv2 
import numpy as np 

src = cv2.imread('../../img/manchu01.jpg', cv2.IMREAD_GRAYSCALE)
tmp_K = cv2.imread('../../img/manchu01K.jpg', cv2.IMREAD_GRAYSCALE)
tmp_L = cv2.imread('../../img/manchu01L.jpg', cv2.IMREAD_GRAYSCALE)
tmp_E = cv2.imread('../../img/manchu01E.jpg', cv2.IMREAD_GRAYSCALE)

# src = cv2.imread('../../img/book/alphabet.bmp', cv2.IMREAD_GRAYSCALE)
# tmp_A = cv2.imread('../../img/book/A.bmp', cv2.IMREAD_GRAYSCALE)
# tmp_S = cv2.imread('../../img/book/S.bmp', cv2.IMREAD_GRAYSCALE)
# tmp_b = cv2.imread('../../img/book/b.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) # Image output

#1
R1 = cv2.matchTemplate(src, tmp_K, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
print('TM_SQDIFF_NORMED: ', minVal, minLoc)

w, h = tmp_K.shape[:2]
cv2.rectangle(dst, minLoc, (minLoc[0] + h, minLoc[1] + w), (255, 0, 0), 2)

#2
R2 = cv2.matchTemplate(src, tmp_L, cv2.TM_CCORR_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R2)
print('TM_CCORR_NORMED: ', maxVal, maxLoc)

w, h = tmp_L.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0] + h, maxLoc[1] + w), (0, 255, 0), 2)

#3
R3 = cv2.matchTemplate(src, tmp_E, cv2.TM_CCOEFF_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R3)
print('TM_CCOEFF_NORMED', maxVal, maxLoc)

w, h = tmp_E.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0] + h, maxLoc[1] + w), (0, 0, 255), 2)

# Multiple Locations (use threshold)
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

# Explanation of this code (zip(*loc[::1]))
# https://stackoverflow.com/questions/56449024/explanation-of-a-few-lines-template-matching-in-python-using-opencv

# red for tmp_E
res_E = cv2.matchTemplate(src, tmp_E, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res_E >= threshold )
for pt in zip(*loc[::-1]):
  cv2.rectangle(dst, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)

# green for tmp_L
res_L = cv2.matchTemplate(src, tmp_L, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res_L >= threshold )
for pt in zip(*loc[::-1]):
  cv2.rectangle(dst, pt, (pt[0] + h, pt[1] + w), (0,255,0), 2)

# blue for tmp_K
res_K = cv2.matchTemplate(src, tmp_K, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res_K >= threshold )
for pt in zip(*loc[::-1]):
  cv2.rectangle(dst, pt, (pt[0] + h, pt[1] + w), (255,0,0), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()