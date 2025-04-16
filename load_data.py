import cv2 as cv
import numpy as np


frames = np.load("data/OnlyHands/photos.npy")
# frame = np.load("data/PAPER/Photo/photo_78.npy")
print(frames.shape)
for i in range(len(frames)):
    cv.imshow("loaded video", frames[i])
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

# cv.imshow("last frame", frame)

cv.waitKey(1000)

cv.destroyAllWindows()