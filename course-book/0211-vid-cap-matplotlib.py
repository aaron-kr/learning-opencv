import cv2
import matplotlib.pyplot as plt

# 1
def handle_key_press(event):
  if event.key == 'escape':
    cap.release()
    plt.close()
def handle_close(event):
  print('Close figure!')
  cap.release()

# 2: Start Program
cap = cv2.VideoCapture(0) # Webcam

plt.ion() # Communication mode settings
fig = plt.figure(figsize=(10,6)) # fig.set_size_inches(10,6)
plt.axis('off')
# ax = fig.gca()
# ax.set_axis_off()
fig.canvas.set_window_title('Video Capture')
fig.canvas.mpl_connect('key_press_event', handle_key_press)
fig.canvas.mpl_connect('close_event', handle_close)

retval, frame = cap.read() # First frame capture
im = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# 3
while True:
  retval, frame = cap.read() # Frame capture
  if not retval:
    break

  # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  fig.canvas.draw()
  # fig.canvas.draw_idle()
  fig.canvas.flush_events() # plt.pause(0.001)

if cap.isOpened():
  cap.release()