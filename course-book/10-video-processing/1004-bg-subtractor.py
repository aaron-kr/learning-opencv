# 1004.py
import cv2
import numpy as np

#1
cap = cv2.VideoCapture('../data/vtest.avi')
if ( not cap.isOpened() ):
  print('Error opening video')

h,w = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

#1
bgMog1 = cv2.createBackgroundSubtractorMOG2()
bgMog2 = cv2.createBackgroundSubtractorMOG2(varThreshold = 25, detectShadows = False)

bgKnn1 = cv2.createBackgroundSubtractorKNN()
bgKnn1 = cv2.createBackgroundSubtractorKNN(dist2Threshold = 1000, detectShadows = False)
# KNN = K-Nearest Neighbors

#2
AREA_TH = 80 # area threshold

def findObjAndDraw(bImg, src):
  res = src.copy()
  bImg = cv2.erode(bImg, None, 5)
  bImg = cv2.dilate(bImg, None, 5)
  bImg = cv2.erode(bImg, None, 7)
  
  contours, _ = cv2.findContours(bImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(src, contours, -1, (255,0,0), 1)
  
  for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > AREA_TH:
      x,y,w,h = cv2.boundingRect(cnt)
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
  
  return res

#3
t = 0
while True:
  ret, frame = cap.read()
  if not ret:
    break
  t += 1
  print('t = ', t)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(frame, (5,5), 0.0)
  
  #2-1
  bImg1 = bgMog1.apply(blur)
  bImg2 = bgMog2.apply(blur)
  bImg3 = bgKnn1.apply(blur)
  bImg4 = bgKnn1.apply(blur)
  dst1 = findObjAndDraw(bImg1, frame)
  dst2 = findObjAndDraw(bImg2, frame)
  dst3 = findObjAndDraw(bImg3, frame)
  dst4 = findObjAndDraw(bImg4, frame)
  
  ## if t == 50:
  cv2.imshow('bImg1', bImg1)
  cv2.imshow('bgMog1', dst1)
  cv2.imshow('bImg2', bImg2)
  cv2.imshow('bgMog2', dst2)
  cv2.imshow('bImg3', bImg3)
  cv2.imshow('bgKnn1', dst3)
  cv2.imshow('bImg4', bImg4)
  cv2.imshow('bgKnn1', dst4)
  
  key = cv2.waitKey(25) #0
  if key == 27:
    break
    
if cap.isOpened():
  cap.release()
cv2.destroyAllWindows()
