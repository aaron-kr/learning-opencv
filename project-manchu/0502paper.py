# 0502.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Manchurian script image source
src = cv2.imread('./img/manchu01.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src',  src)
height, width = src.shape
print("IMAGE: width = ", width, "height = ", height)

# Create an array with the data from the cols of the image
cols = np.full(width, 0)
# Create binary image (only 1s and 0s) using threshold
ret, bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# inverse binary image (black bg, white txt)
ibin = cv2.bitwise_not(bin)
#area = bin[0:1000,0:100]
#print(area)
# cv2.imshow('dst Otsu+Binary inverted',  ibin)
# print(cols[0])

"""
Let's create and call some functions to perform each task. Maybe...
1. findLines() : Divide the image into columns of text
2. findWords() : Divide the columns of text into words
3. findLetters() : Divide the words into individual letters
"""

"""
1. findLines() : Divide the image into columns of text
"""
# For every col, find anything with data (a pixel of the script)
for i in range(width):
    cols[i] = cv2.countNonZero(ibin[:,i])

# Var to hold num of font areas (vertical lines of text) based on the script
n_fontarea = 0

# Determine font areas by checking where a non-zero col ends and a zero col begins
for i in range(width - 1):
    if cols[i] > 0 and cols[i+1] == 0: # here, our script ends, and whitespace begins
        n_fontarea = n_fontarea + 1 # so, it's the end of a n_fontarea (+1)

# Tell me how many font areas there are (i.e. how many vertical lines of text)
print("Number of font areas = ", n_fontarea)

# Make a copy of the cols to manipulate it
bcols = cols.copy()

for i in range(width):
    if cols[i] > 0: # if some data exists in this col
        bcols[i] = 1 # then set bcols at the same location to 1 (binary)

# Setup image cut points (x axis value) for +1 greater than the number of fontareas 
# so that we can cut AROUND each column of text.
# i.e. 13 columns of text requires 14 lines (cut points) to divide them 
cutXpoints = np.full(n_fontarea + 1, 0)
# print('cutXpoints = ', cutXpoints) # cutXpoints =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# Initialize variables
j = 0 # cutXpoints counter (we have 14 cut points)
startpoint = 0 # start at the beginning of the image (col 0)

# Loop to determine and set our cutXpoints (where to cut the image for each column of text)
for i in range(1, width): # start at 1, end at width
    # The first case is the END of the image (width - 1) i.e. if 458 == 458 
    if i == width - 1:
        endpoint = i
        cutXpoints[j] = (startpoint + endpoint) // 2 # / is floating point division, // is integer division (floor - rounding down)
    # Case 2 is the START of a cut point, i.e. the first col is all 0s (whitespace) and the second col is 1 (script)
    elif bcols[i-1] == 0 and bcols[i] == 1:
        endpoint = i - 1 # don't cut off the script, cut outside it
        cutXpoints[j] = (startpoint + endpoint) // 2
        j = j + 1 # increment cutXpoints counter
    # Case 3 is the END of a cut point, i.e. the first col has script (1), and the second col is all 0s (whitespace)
    elif bcols[i-1] == 1 and bcols[i] == 0:
        startpoint = i # in this case, adjust the startpoint to the current column

# Confirm our points
print("cutXpoints = ", cutXpoints)
# plt.plot(cols)
# plt.show()

# Now, using the cutXpoints we determined, cut out and display one column of text (change array values)
fcarea = bin[0:height, cutXpoints[0]:cutXpoints[1]]
# print('type of fcarea = ', type(fcarea))

# script_line = np.array(np.ndarray)
# # print('script_line = ', script_line)
# for i in range(0, n_fontarea - 1):
#     j = i + 1
#     script_line[i] = bin[0:height, cutXpoints[i]:cutXpoints[j]]
# cv2.imshow('font area', fcarea)

"""
2. findWords() : Divide the columns of text into words
"""
def findWords(line):
    rows = np.full(height, 0)
    iline = cv2.bitwise_not(line)

    for i in range(height):
        rows[i] = cv2.countNonZero(iline[i,:])

    # Var to hold num of word areas in column 1
    n_wordarea = 0

    for i in range(height - 1):
        if rows[i] > 0 and rows[i+1] == 0:
            n_wordarea = n_wordarea + 1

    print("Number of words in line = ", n_wordarea)

    brows = rows.copy()

    for i in range(height):
        if rows[i] > 0:
            brows[i] = 1

    # print('brows = ', brows)

    cutYpoints = np.full(n_wordarea + 1, 0)
    # print('cutYpoints = ', cutYpoints)

    j = 0
    startpoint = 0

    for i in range(1, height):
        if i == height - 1:
            endpoint = i
            cutYpoints[j] = (startpoint + endpoint) // 2
        elif brows[i-1] == 0 and brows[i] == 1:
            endpoint = i - 1
            cutYpoints[j] = (startpoint + endpoint) // 2
            j = j + 1
        elif brows[i-1] == 1 and brows[i] == 0:
            startpoint = i

    print("cutYpoints = ", cutYpoints)
    plt.plot(rows)
    plt.show()

    # Create a new image with the cut points
    fcword = bin[cutYpoints[0]:cutYpoints[1], cutXpoints[0]:cutXpoints[1]]
    # cv2.imshow('word', fcword)
    cv2.imwrite('./img/manchu01b.jpg',fcword)

    return fcword

word = findWords(fcarea)

def findLetter(word):
    height, width = word.shape
    print("WORD: width = ", width, "height = ", height)

    cv2.imshow('Letter finding word', word)
    w_rows = np.full(height, 0)

    ret, w_bin = cv2.threshold(word, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    w_ibin = cv2.bitwise_not(w_bin)

    for i in range(height):
        w_rows[i] = cv2.countNonZero(w_ibin[i,:])
    
    n_letters = 0

    print("wordRows = ", w_rows)

    for i in range(1, height - 1):
        if w_rows[i-1] >= 4 and w_rows[i] <= 4 and w_rows[i+1] >= 4:
            print("checking: ", w_rows[i-1], ", ", w_rows[i], ", ", w_rows[i+1])
            n_letters = n_letters + 1
    
    print("Number of letters in word = ", n_letters)

    arows = w_rows.copy()

    for i in range(height):
        if w_rows[i] > 0:
            arows[i] = 1

    cutSubYpoints = np.full(n_letters + 1, 0)

    j = 0
    startpoint = 0

    for i in range(1, height):
        if i == height - 1:
            endpoint = i
            cutSubYpoints[j] = (startpoint + endpoint) // 2
        elif arows[i-1] == 0 and arows[i] == 1:
            endpoint = i - 1
            cutSubYpoints[j] = (startpoint + endpoint) // 2
            j = j + 1
        elif arows[i-1] == 1 and arows[i] == 0:
            startpoint = i
    
    print("cutSubYpoints = ", cutSubYpoints)
    plt.plot(w_rows)
    plt.show()

    # Create a new image with the cut points
    fletter = bin[cutSubYpoints[0]:cutSubYpoints[1], 0:width]
    cv2.imshow('letter', fletter)

    return fletter

findLetter(word)
# Determine word areas by checking where a non-zero col ends and a zero col begins
# for i in range(height - 1):
    # if 

# So, now that we have columns of text, lets find cutYpoints for every word
# let's start with the first column of text (and build up later)
# cutYpoints = 


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
