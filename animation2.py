import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Video:
  def __init__(self, device = 0):
    self.cap = cv2.VideoCapture(device)
    self.retval, self.frame = self.cap.read()
    self.im = plt.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
    print('start capture...')

  def updateFrame(self, k):
    self.retval, self.frame = self.cap.read()
    self.im = plt.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
    ## return self.im,

  def close(self):
    if self.cap.isOpened():
      self.cap.release()
    print('finish capture')

# Program Start
fig = plt.figure()
fig.canvas.set_window_title('Video Capture')
plt.axis('off')

camera = Video()
## camera = Video('./data/record0.mp4')
ani = animation.FuncAnimation(fig, camera.updateFrame, interval = 50)
plt.show()
camera.close()