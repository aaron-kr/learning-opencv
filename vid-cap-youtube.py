import cv2, pafy # https://pythonhosted.org/Pafy/

url = 'https://www.youtube.com/watch?v=5UirhqmmXlI'
video = pafy.new(url)
print('title = ', video.title)
print('video.rating = ', video.rating)
print('video.duration = ', video.duration)
print(video)

best = video.getbest(preftype = 'webm', ftypestrict = False) # 'mp4', '3gp'
# ftypestrict = False returns a different preftype if one with a higher res exists
print('best.resolution = ', best.resolution)

cap = cv2.VideoCapture(best.url)
# cap = cv2.VideoCapture(url)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size = ', frame_size)

while True:
  retval, frame = cap.read() # Frame capture
  if not retval:
    break

  cv2.imshow('frame', frame)

  # New stuff
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 100, 200)
  cv2.imshow('edges', edges)

  key = cv2.waitKey(25)
  if key == 27: # Esc
    break

if cap.isOpened():
  cap.release()

cv2.destroyAllWindows()