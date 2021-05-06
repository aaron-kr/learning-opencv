# 0919.py
import cv2 

src1 = cv2.imread('../../img/stitch1.jpg')
src2 = cv2.imread('../../img/stitch2.jpg')
src3 = cv2.imread('../../img/stitch3.jpg')
src4 = cv2.imread('../../img/stitch4.jpg')

stitcher = cv2.Stitcher.create()
status, dst = stitcher.stitch((src1, src2, src3, src4))
cv2.imwrite('../../img/stitch_out.jpg', dst)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()