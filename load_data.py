import cv2 as cv
import numpy as np

frames = np.load("./data/SCISSORS/video.npy")

for i in range(len(frames)):
    cv.imshow("loaded video", frames[i])
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()