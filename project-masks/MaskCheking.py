#1119.py #1120.py

import cv2
import numpy as np
# import cv2, pafy

faceCascade= cv2.CascadeClassifier(
      './haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(
    './haarcascades/haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_mcs_nose.xml")  # 코찾기 haar 파일
mouthCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_mcs_mouth.xml")  # 입찾기 haar 파일


cap = cv2.VideoCapture(0)       # Web Camera ON
#ret, frame = cap.read()

##image = input("input image name: ")
##src = cv2.imread(image) #persons6
##dst = src.copy()
##gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#url = 'https://www.youtube.com/watch?v=S_0ikqqccJs'
#video = pafy.new(url)
#print('title = ', video.title)

#best = video.getbest(preftype='mp4')
#print('best.resolution', best.resolution)
n_faces = 0
#cap=cv2.VideoCapture(best.url)
#n_eyes = 0
while(True):
        retval, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not retval:
                break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ##dst = frame.copy()
        faces = faceCascade.detectMultiScale(gray, minSize=(50,50)) ## limit size of face
        n_faces = len(faces)
        n_eyes = n_noses = n_mouthes = 0
        ##print("Number of faces = ",faces)
        for (x, y, w, h) in faces:
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w] ##dst->frame
            eyes = eyeCascade.detectMultiScale(roi_gray, maxSize=(90,90), minSize=(50,50)) ## limit the size of eyes
            ##print("Number of eyes = ", eyes)
            ##n_eyes = len(eyes)
            ##if  n_eyes == 2:

            # Face Rectangle
            cv2.rectangle(frame, (x,y),(x+w, y+h),(233,31,199), 2) # purple

            for (ex,ey,ew,eh) in eyes:
                if ey < (h//2): ## find the eyes in the upper part of faces

                    # Eyes Rectangle
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255, 181, 5),2) # blue
                    n_eyes += 1


            mouthes = mouthCascade.detectMultiScale(roi_gray, maxSize=(90,90), minSize=(50,50)) ## limit the size of mouthes
            for (mx,my,mw,mh) in mouthes:
                if my > (h//2): ## find mouthes in the lower part of faces
                    cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,220,255),2)   

            noses = noseCascade.detectMultiScale(roi_gray, maxSize=(90,90), minSize=(50,50)) ## limit the size of noese   
            for (nx,ny,nw,nh) in noses:
                if ny <= (2*h//3) and ny >= (h//3) :    ##find the noses in the middle part of faces
                    cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(59, 199, 245),2)
                    n_noses += 1

            if n_faces == 0 or n_noses > 0 or n_mouthes > 0:
                print("Wear a mask, please!")
            else :
                print("Thank you for the mask!")
                    
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1000)
        if key == 27: # Esc                           
            break
            cv2.destroyAllWindows()
        elif key == ord('s'): # 's' key
            filename = input('Image filename to save : ')
            cv2.imwrite(filename,frame)
cv2.destroyAllWindows()
