import cv2
import numpy as np

width, height = 512, 512
x, y, R = 256, 256, 50
direction = 0  # right

while True:
    key = cv2.waitKeyEx(30)
    if key == 0x1B:
        break

    # Direction keys
    elif key == 0x27:  # right
        direction = 0
    elif key == 0x28:  # down
        direction = 1
    elif key == 0x25:  # left
        direction = 2
    elif key == 0x26:  # up
        direction = 3

    # Direction movement
    if direction == 0:  # right
        x += 10
    elif direction == 1:  # down
        y += 10
    elif direction == 2:  # left
        x -= 10
    elif direction == 3:  # up
        y -= 10

    # Checking the borders
    if x < 0 - R:
        x = width + R
    if x > width + R:
        x = 0 - R
    if y < 0 - R:
        y = height + R
    if y > height + R:
        y = 0 - R

    # Erase and redraw
    img = np.zeros((width, height, 3), np.uint8) + 255  # erase the old one
    cv2.circle(img, (x, y), R, (0, 0, 255), -1)
    cv2.imshow('img', img)

# cv2.waitKey(0)  #wait for any key
cv2.destroyAllWindows()
