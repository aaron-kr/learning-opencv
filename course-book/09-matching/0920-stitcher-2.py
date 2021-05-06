# 0920.py
import cv2 

src1 = cv2.imread('../../img/stitch1.jpg')
src2 = cv2.imread('../../img/stitch2.jpg')
src3 = cv2.imread('../../img/stitch3.jpg')
src4 = cv2.imread('../../img/stitch4.jpg')

stitcher = cv2.Stitcher.create()
status, dst2 = stitcher.stitch((src1, src2))
status, dst3 = stitcher.stitch((src2, src3))
status, dst4 = stitcher.stitch((src3, src4))

cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()