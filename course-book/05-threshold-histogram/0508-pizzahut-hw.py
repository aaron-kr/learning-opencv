# 0508.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

# This is BGR
src = cv2.imread('../../img/pizzahut.jpg')
b, g, r = cv2.split(src)
dst_b = cv2.equalizeHist(b)
dst_r = cv2.equalizeHist(r)
dst_g = cv2.equalizeHist(g)

dst_bgr = cv2.merge([dst_b, dst_g, dst_r])
cv2.imshow('dst_bgr', dst_bgr)

# Also do HSV
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
dst_h = cv2.equalizeHist(h)
dst_s = cv2.equalizeHist(s)
dst_v = cv2.equalizeHist(v)

dst_hsv = cv2.merge([dst_h, dst_s, dst_v])
cv2.imshow('dst_hsv', dst_hsv)

# Also try YCrCb
yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y, Cr, Cb = cv2.split(yCrCb)
dst_y = cv2.equalizeHist(y)
dst_Cr = cv2.equalizeHist(Cr)
dst_Cb = cv2.equalizeHist(Cb)

dst_yCrCb = cv2.merge([dst_y, dst_Cr, dst_Cb])
cv2.imshow('dst_yCrCb', dst_yCrCb)

# dst = cv2.equalizeHist(src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

plt.title('Grayscale histogram')

# Histograms
hist_src = cv2.calcHist(images = [src], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.plot(hist_src, color = 'gray', label = 'hist_src in src')

hist_bgr = cv2.calcHist(images = [dst_bgr], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.plot(hist_bgr, color = 'blue', alpha = 0.7, label = 'hist_bgr in dst_bgr')

hist_hsv = cv2.calcHist(images = [dst_hsv], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.plot(hist_hsv, color = 'purple', alpha = 0.7, label = 'hist_hsv in dst_hsv')

hist_yCrCb = cv2.calcHist(images = [dst_yCrCb], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.plot(hist_yCrCb, color = 'yellow', alpha = 0.7, label = 'hist_yCrCb in dst_yCrCb')

plt.legend(loc = 'best')
plt.show()