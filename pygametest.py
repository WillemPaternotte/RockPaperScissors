import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv.imshow("webcam", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap objectq
vid.release()
# Destroy all the windows
cv.destroyAllWindows()