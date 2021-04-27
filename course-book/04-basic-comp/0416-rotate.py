# 0416.py
import cv2
src = cv2.imread('../../img/link-512.jpg')

rows, cols, channels = src.shape
M1 = cv2.getRotationMatrix2D((rows / 2, cols / 2), 45, 0.5)
M2 = cv2.getRotationMatrix2D((rows / 2, cols / 2), -45, 1.0)

dst1 = cv2.warpAffine(src, M1, (rows, cols))
dst2 = cv2.warpAffine(src, M2, (rows, cols))

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()