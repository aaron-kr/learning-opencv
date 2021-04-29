# 0612.py
import cv2 
import numpy as np 

src = cv2.imread('./img/manchu01.jpg', cv2.IMREAD_GRAYSCALE)
tmp_K = cv2.imread('./img/man_k.jpg', cv2.IMREAD_GRAYSCALE)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) # Image output

#1
R1 = cv2.matchTemplate(src, tmp_K, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
print('TM_SQDIFF_NORMED: ', minVal, minLoc)

w, h = tmp_K.shape[:2]
cv2.rectangle(dst, minLoc, (minLoc[0] + h, minLoc[1] + w), (255, 0, 0), 2)

# Multiple Locations (use threshold)
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

# Explanation of this code (zip(*loc[::1]))
# https://stackoverflow.com/questions/56449024/explanation-of-a-few-lines-template-matching-in-python-using-opencv

# blue for tmp_K
res_K = cv2.matchTemplate(src, tmp_K, cv2.TM_CCOEFF_NORMED)
# res_K = cv2.matchTemplate(src, tmp_K, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(res_K)
print('TM_CCOEFF_NORMED: ', minVal, minLoc)

threshold = 0.8
loc = np.where( res_K >= threshold )

count = 0
for pt in zip(*loc[::-1]):
  cv2.rectangle(dst, pt, (pt[0] + h, pt[1] + w), (255,0,0), 2)
  count += 1

print('count = ', count)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()