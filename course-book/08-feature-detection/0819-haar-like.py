# 0819.py
import cv2
import numpy as np

#1
def rectSum(sumImg, rect):
	x,y,w,h = rect
	a = sumImg[y,x]
	b = sumImg[y,x+w]
	c = sumImg[y+h,x]
	d = sumImg[y+h,x+w]
	return a + d - b - c

#2
def compute_Haar_feature1(sumImg, rect):
	x,y,w,h = rect
	## print(x,y,w,h)
	s1 = rectSum(sumImg, (x,y,w,h))
	s2 = rectSum(sumImg, (x+w,y,w,h))
	## print('s1 =', s1)
	## print('s2 = ', s2)
	return s1-s2

def compute_Haar_feature2(sumImg, rect):
	x,y,w,h = rect
	s1 = rectSum(sumImg, (x,y,w,h))
	s2 = rectSum(sumImg, (x,y+h,w,h))
	return s2 - s1

def compute_Haar_feature3(sumImg, rect):
	x,y,w,h = rect
	s1 = rectSum(sumImg, (x,y,w,h))
	s2 = rectSum(sumImg, (x+w,y,w,h))
	s3 = rectSum(sumImg, (x+2*w,y,w,h))
	return s1 - s2 + s3
	
def compute_Haar_feature4(sumImg, rect):
	x,y,w,h = rect
	s1 = rectSum(sumImg, (x,y,w,h))
	s2 = rectSum(sumImg, (x,y+h,w,h))
	s3 = rectSum(sumImg, (x,y+2*h,w,h))
	return s1 - s2 + s3

def compute_Haar_feature5(sumImg, rect):
	x,y,w,h = rect
	s1 = rectSum(sumImg, (x,y,w,h))
	s2 = rectSum(sumImg, (x+w,y,w,h))
	s3 = rectSum(sumImg, (x,y+h,w,h))
	s4 = rectSum(sumImg, (x+w,y+h,w,h))
	return s1 + s4 - s2 - s3

#3
A = np.arange(1,6*6+1).reshape(6,6).astype(np.uint8)
print('A = ', A)

h,w = A.shape
sumA = cv2.integral(A)
print('sumA = ', sumA)

#4
f1 = compute_Haar_feature1(sumA, (0,0,w//2,h)) # 3,6
print('f1 = ', f1)

#5
f2 = compute_Haar_feature2(sumA, (0,0,w,h//2)) # 6,3
print('f2 = ', f2)

#6
f3 = compute_Haar_feature3(sumA, (0,0,w//3,h)) # 2,6
print('f3 = ', f3)

#7
f4 = compute_Haar_feature4(sumA, (0,0,w,h//3)) # 6,2
print('f4 = ', f4)

#8
f5 = compute_Haar_feature5(sumA, (0,0,w//2,h//2)) # 3,3
print('f5 = ', f5)