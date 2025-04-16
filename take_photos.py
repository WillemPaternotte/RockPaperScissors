import cv2 as cv
import numpy as np
import glob
import os

images = []


def get_last_photo_index(name):
    list_of_files = glob.glob('./data/' + name + '/Photo/*')  # * means all if need specific format then *.csv
    if len(list_of_files) <= 0:
        return -1
    latest_file = max(list_of_files, key=os.path.getctime)
    last_index = latest_file[20 + len(name):]
    return int(last_index[:-4])

if __name__ == "__main__":
    last_index = get_last_photo_index("EMPTY")
    vid = cv.VideoCapture(0)

    ret, frame = vid.read()
    last_frame = cv.resize(frame, (320, 180))

    running = True
    while running:
        ret, frame = vid.read()
        last_frame = cv.resize(frame, (320, 180))

        cv.imshow("video", last_frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('s'):
            last_index += 1
            np.save("./data/EMPTY/Photo/photo_" + str(last_index) + ".npy", last_frame)
            print("saved photo: ", last_index)
        if key == ord('q'):
            running = False

    cv.destroyAllWindows()
