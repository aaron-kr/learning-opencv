import cv2 

cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size', frame_size)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX') # ('D','I','V','X')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out0 = cv2.VideoWriter('../data/record0.mp4', fourcc, 20.0, frame_size)
out1 = cv2.VideoWriter('../data/record1.mp4', fourcc, 20.0, frame_size, isColor = False )

while True:
  retval, frame = cap.read()
  if not retval:
    break

  # out0 (color)
  out0.write(frame)

  # out1 (gray)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  out1.write(gray)

  cv2.imshow('frame', frame)
  cv2.imshow('gray', gray)

  key = cv2.waitKey(25)
  if key == 27:
    break

cap.release()
out0.release()
out1.release()
cv2.destroyAllWindows()