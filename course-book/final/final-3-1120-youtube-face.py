# 1120.py
'''
pip install youtube_dl
pip install pafy
'''
import numpy as np
import cv2, pafy

# define / load haarcascades
faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')
mouthCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_mouth.xml')

# define / load YouTube video
url = 'https://youtube.com/watch?v=S_0ikqqccJs&t=5'
video = pafy.new(url)
print('title = ', video.title)

best = video.getbest(preftype = 'webm', ftypestrict = False) # .mp4, .3gp
print('best.resolution = ', best.resolution) # error here - check how we fixed this in previous chapter with YouTube

cap = cv2.VideoCapture(best.url)

# Loop through YouTube video frames and do some work
while(True):
  retval, frame = cap.read()
  if not retval:
    break
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(gray) # minSize = (50,50)
  eyes = eyeCascade.detectMultiScale(gray) 
  noses = noseCascade.detectMultiScale(gray)
  mouths = mouthCascade.detectMultiScale(gray)

  if ( len(faces) > 0 and len(eyes) > 1 and len(eyes) % 2 == 0 and len(noses) > 0 and len(mouths) > 0):

    # Faces: blue
    for (x,y,w,h) in faces:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    # Eyes: green
    # Search area: ONLY upper half of the face
    # Don't draw if not 2 eyes
    for (x,y,w,h) in eyes:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Noses: yellow
    # Limited lower 1/4 face area?
    for (x,y,w,h) in eyes:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    # Mouths: red
    # Search area: ONLY lower half of the face
    for (x,y,w,h) in eyes:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

  cv2.imshow('frame', frame)

  key = cv2.waitKey(25)
  if key == 27: # Esc
    break

cv2.destroyAllWindows()