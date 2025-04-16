import cv2 as cv
import numpy as np
import random
import glob
from data_augmentation import  load_data, mirror_data
from sklearn.model_selection import train_test_split


def crop_around_box(image, box):
    x,y,w,h = box
    largest_side = int(np.max((w,h)))
    middlepoint = (int(x+w/2), int(y+h/2))
    if largest_side>180:
        largest_side = 180
    topleft = [int(middlepoint[0]-largest_side/2),int(middlepoint[1]-largest_side/2)]
    bottomright = [int(middlepoint[0] + largest_side / 2), int(middlepoint[1] + largest_side / 2)]
    if topleft[0]<0:
        topleft[0] = 0
        bottomright[0] = largest_side
    if topleft[1]<0:
        topleft[1] = 0
        bottomright[1] = largest_side
    if bottomright[0]>320:
        topleft[0] = 320-largest_side
        bottomright[0] = 320
    if bottomright[1]>180:
        topleft[1] = 180 - largest_side
        bottomright[1] = 180
    if topleft[0] == bottomright[0]:
        bottomright[0] += 1
    if topleft[1] == bottomright[1]:
        bottomright[1] += 1
    return cv.resize(image.copy()[topleft[1]:bottomright[1], topleft[0]:bottomright[0]], (64,64))

if __name__ == "__main__":
    X0, y0 = load_data('./data/ROCK/PHOTO/*')
    X1, y1 = load_data('./data/PAPER/PHOTO/*')
    X2, y2 = load_data('./data/SCISSORS/PHOTO/*')
    print(X0[0].shape)

    X_train = []
    y_train = []
    for i in range(len(X0)):
        X_train.append(crop_around_box(X0[i], y0[i]))
        y_train.append([1,0,0])
    for i in range(len(X1)):
        X_train.append(crop_around_box(X1[i], y1[i]))
        y_train.append([0,1,0])
    for i in range(len(X2)):
        X_train.append(crop_around_box(X2[i], y2[i]))
        y_train.append([0,0,1])

    # np.save('./data/OnlyHands/photos', X_train)
    # np.save('./data/OnlyHands/labels', y_train)