# 1001.py
import cv2
import numpy as np

#1
cap = cv2.VideoCapture('../data/vtest.avi')
if ( not cap.isOpened() ):
    print('Error opening video.')

h,w = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

acc_gray = np.zeros(shape = (h,w), dtype = np.float32)
acc_bgr = np.zeros(shape = (h,w,3), dtype = np.float32)
t = 0

#2
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t += 1
    print('t = ', t)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #2-1
    cv2.accumulate(gray, acc_gray)
    avg_gray = acc_gray / t 
    dst_gray = cv2.convertScaleAbs(avg_gray)

    #2-2
    cv2.accumulate(frame, acc_bgr)
    avg_bgr = acc_bgr / t
    dst_bgr = cv2.convertScaleAbs(avg_bgr)

    cv2.imshow('frame', frame)
    cv2.imshow('dst_gray', dst_gray)
    cv2.imshow('dst_bgr', dst_bgr)
    key = cv2.waitKey(20)
    if key == 27:
        break

#3
if cap.isOpened(): cap.release()
cv2.imwrite('../data/avg_gray.png', dst_gray)
cv2.imwrite('../data/avg_bgr.png', dst_bgr)
cv2.destroyAllWindows()