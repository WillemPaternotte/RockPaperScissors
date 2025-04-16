import cv2 as cv
import random

import numpy as np

cap = cv.VideoCapture('./data/EMPTY/WIN_20250411_10_48_40_Pro.mp4')

framecounter = 0
savecounter = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.resize(frame,(320,180))
    framecounter+=1
    print(framecounter)
    if framecounter > 1600 and random.random() < 0.05:
        np.save("./data/EMPTY/Photo/photo_"+str(savecounter)+".npy", frame)
        savecounter+=1

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()