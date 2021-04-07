# 0502.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Manchurian script image source
src = cv2.imread('./img/manchu01.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src',  src)
height,width = src.shape
print("width = ", width, "height = ", height)

# Create an array with the data from the cols of the image
cols = np.full(width,0)
# Create binary image (only 1s and 0s) using threshold
ret, bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# inverse binary image (black bg, white txt)
ibin = cv2.bitwise_not(bin)
#area = bin[0:1000,0:100]
#print(area)
cv2.imshow('dst Otsu+Binary inverted',  ibin)
# print(cols[0])

# For every col, find anything with data (a pixel of the script)
for i in range(width):
    cols[i] = cv2.countNonZero(ibin[:,i])

# Var to hold num of font areas (vertical lines of text) based on the script
n_fontarea = 0

# Determine font areas by checking where a non-zero col ends and a zero col begins
for i in range(width-1):
    if cols[i] > 0 and cols[i+1] == 0: # here, our script ends, and whitespace begins
        n_fontarea = n_fontarea + 1 # so, it's the end of a n_fontarea (+1)

# Tell me how many font areas there are (i.e. how many vertical lines of text)
print("Number of font area = ", n_fontarea)

# Make a copy of the cols to manipulate it
bcols = cols.copy()
for i in range(width):
    if cols[i] > 0:
        bcols[i]=1
cutXpoints = np.full(n_fontarea+1,0)
j=0
startpoint = 0
for i in range(1,width):
    if i== width-1:
        endpoint = i
        cutXpoints[j]= (startpoint + endpoint)//2
    elif bcols[i-1] == 0 and bcols[i] == 1 :
        endpoint = i-1
        cutXpoints[j]= (startpoint + endpoint)//2
        j=j+1
    elif bcols[i-1] == 1 and bcols[i] == 0:
        startpoint = i
print(cutXpoints)
plt.plot(cols)
plt.show()

fcarea = bin[0:height, cutXpoints[0]:cutXpoints[1]]
cv2.imshow('font area', fcarea)


# for i in range(width):
#    print(i, " = ", cols[i])

# # print([rows])
# roi = cv2.selectROI(src)
# print('roi = ', roi)
# bimg = src[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
# ret, dst = cv2.threshold(src, 0, 255,
#                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('dst Otsu+Binary',  dst)

# dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                            cv2.THRESH_BINARY, 51, 7)
# cv2.imshow('dst2 AdaptiveThreshMeanC+Binary',  dst2)

# dst3 = cv2.adaptiveThreshold(bimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv2.THRESH_BINARY, 51, 7)
# cv2.imshow('dst3 AdaptiveThreshGaussianC+Binary',  dst3)
# cv2.imwrite('./img/manchu01b.jpg',dst3)


cv2.waitKey()    
cv2.destroyAllWindows()
