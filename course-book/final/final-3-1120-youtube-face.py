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

  faces = faceCascade.detectMultiScale(gray, minNeighbors = 5, minSize = (50,50)) # minSize = (50,50)

  if ( len(faces) > 0):

    # Faces: blue
    for (x,y,w,h) in faces:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

      face = gray[y:y+h,x:x+w]
      faceHeight = face.shape[0]
      faceWidth = face.shape[1]

      # Let's say the face is 200px high, and we want to divide it into four sections
      # topOfFace = 0, bottomOfFace = faceHeight = 200
      # we want to divide at every 50px
      # oneofFour = 1 * faceHeight//4
      # twoofFour = 2 * faceHeight//4
      # threeofFour = 3 * faceHeight//4
      # fourofFour = 4 * faceHeight//4

      # Eyes: GREEN
      # Search area: ONLY upper half of the face
      eyeSearchStart = 0
      eyeSearchEnd = faceHeight//2

      eyeSearchArea = face[eyeSearchStart:eyeSearchEnd,0:faceWidth] # from y (0) to half height ((y+h)//2)
      eyes = eyeCascade.detectMultiScale(eyeSearchArea, minNeighbors = 15, minSize = (20,20))

      # Don't draw if not 2 eyes
      if (len(eyes) >= 2):
        for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,255,0), 2)

      # Noses: YELLOW
      # Limited lower 1/4 face area?
      noseSearchStart = faceHeight//2
      noseSearchEnd = 3 * faceHeight//4

      noseSearchArea = face[noseSearchStart:noseSearchEnd,0:faceWidth]
      noses = noseCascade.detectMultiScale(noseSearchArea, minNeighbors = 5, minSize = (20,20))
      for (nx,ny,nw,nh) in noses:
        cv2.rectangle(frame, (x+nx,y+ny+noseSearchStart), (x+nx+nw,y+ny+nh+noseSearchStart), (0,255,255), 2)

      # Mouths: RED
      # Search area: ONLY lower 1/3 of the face
      mouthSearchStart = 2 * faceHeight//3
      mouthSearchEnd = faceHeight

      mouthSearchArea = face[mouthSearchStart:mouthSearchEnd,0:faceWidth]
      mouths = mouthCascade.detectMultiScale(mouthSearchArea, minNeighbors = 5, minSize = (20,20)) # , minSize = (20,20), maxSize = (50,50)
      for (mx,my,mw,mh) in mouths:
        cv2.rectangle(frame, (x+mx,y+my+mouthSearchStart), (x+mx+mw,y+my+mh+mouthSearchStart), (0,0,255), 2)

  cv2.imshow('frame', frame)

  key = cv2.waitKey(25)
  if key == 27: # Esc
    break

cv2.destroyAllWindows()