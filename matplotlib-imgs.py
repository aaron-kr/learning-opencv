import cv2
from matplotlib import pyplot as plt

path = './img/'
imageFile = path + 'basketball.jpg'

## MATPLOTLIB 1 : BGR
## Matplotlib wants to show in RGB though...
imgBGR = cv2.imread(imageFile) # cv2.IMREAD_COLOR
# plt.axis('off')
# plt.imshow(imgBGR)
# plt.show()

# So, let's convert the color properly (Matplotlib = cvtColor())
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
# plt.imshow(imgRGB) # This is using Matplotlib's imshow() so it needs conversion (line 12)
# plt.show()

# About BGR to RGB
# https://stackoverflow.com/questions/50963283/python-opencv-imshow-doesnt-need-convert-from-bgr-to-rgb
# OpenCV imread, imwrite, imshow = BGR
# Matplotlib imshow = convert to RGB

# =================================================

## MATPLOTLIB 2 : Grayscale
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE) # 0
# plt.axis('off')
# plt.imshow(imgGray, cmap = 'gray', interpolation = 'bicubic')
# plt.show()

# =================================================

## MATPLOTLIB 3 : Resize
# plt.figure(figsize = (6,6))
# plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
# plt.imshow(imgGray, cmap = 'gray')
# ## plt.axis('tight')
# plt.axis('off')
# plt.savefig(path + 'bball-gray.png')
# plt.show()

# =================================================

## MATPLOTLIB 4 : Multiple images
imgBGR1 = cv2.imread( path + 'basketball.jpg' )
imgBGR2 = cv2.imread( path + 'spirit-week.jpg' )
imgBGR3 = cv2.imread( path + 'gpa-newsletter-nov-2019.png' )
imgBGR4 = cv2.imread( path + 'bball-gray.png' )

# Convert color BGR -> RGB
imgRGB1 = cv2.cvtColor( imgBGR1, cv2.COLOR_BGR2RGB )
imgRGB2 = cv2.cvtColor( imgBGR2, cv2.COLOR_BGR2RGB )
imgRGB3 = cv2.cvtColor( imgBGR3, cv2.COLOR_BGR2RGB )
imgRGB4 = cv2.cvtColor( imgBGR4, cv2.COLOR_BGR2RGB )

fig, ax = plt.subplots(2, 2, figsize = (10, 10), sharey = True )
fig.canvas.set_window_title( '4-pic Grid' )

# Img 1
ax[0][0].axis('off')
ax[0][0].imshow(imgRGB1, aspect = 'equal')

# Img 2
ax[0][1].axis('off')
ax[0][1].imshow(imgRGB2, aspect = 'equal')

# Img 3
ax[1][0].axis('off')
ax[1][0].imshow(imgRGB3, aspect = 'equal') # 'auto' = stretched

# Img 4
ax[1][1].axis('off')
ax[1][1].imshow(imgRGB4, aspect = 'equal')

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0.05, hspace = 0.05)
# plt.savefig(path + 'grid-pics.jpg', bbox_inches = 'tight')
plt.show()