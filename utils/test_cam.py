import cv2
import numpy as np
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    c = cv2.waitKey(30)

    if c == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# import cv2
# import imutils
#
# cap = cv2.VideoCapture(-1)  # video capture source camera (Here webcam of laptop)
# ret, frame = cap.read()  # return a single frame in variable `frame`
#
#
# while (True):
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     (grabbed, frame) = cap.read()
#     frame = imutils.resize(frame, width=400)
#     cv2.imshow('img1', frame)  # display the captured image
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # save on pressing 'y'
#         cv2.imwrite('capture.png', frame)
#         cv2.destroyAllWindows()
#         break
#
# cap.release()