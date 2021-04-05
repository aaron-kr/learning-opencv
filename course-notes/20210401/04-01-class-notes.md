# REVIEW from last week

img[100,200] = [y,x] = [rows, cols] - this is how to think about that

hsv = best for skin color (what about black people?)
- esp 'h' - hue

**For test**

* take Hanbat Uni logo and overlay it over an image (like 0418.py)
* overlay two images atop each other (baboon.jpg over lena.jpg)

PCA = Principle Component Analysis

* [link](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [step-by-step](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

# NOTES Today

For threshold - find the slump in the histogram
0 min, 255 max, basic thresh = 128, but sometimes not depending on the image histogram - too dark, too light, etc

Values in the threshold - sometimes can find it - but cv2.THRESH_OTSU (made by Japanese guy) is designed to help find those.

0503.py
cv2.calcHist()
ranges=[0,8] - matrix 4x4 - finds values 0,1 then 2,3 then 4,5 then 6,7 (check the PPT again...)

pizzahut - homework = 0502.py cv2.adpativeThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 7) # try 51 as 31 and see it's worse - this is a process of trial-and-error