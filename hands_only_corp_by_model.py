import time

import cv2 as cv
import numpy as np
import pygame
from enum import Enum
import glob
import tensorflow as tf
from model_test import iou_loss, iou_metric
from croparoundbox import crop_around_box
from data_augmentation import load_data, mirror_data

class Hand(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


if __name__ == "__main__":
    reg_model = tf.keras.models.load_model("model_saves/my_model_good.keras", custom_objects={'iou_loss': iou_loss, 'iou_metric': iou_metric})

    X0, y0 = load_data('./data/ROCK/PHOTO/*')
    X1, y1 = load_data('./data/PAPER/PHOTO/*')
    X2, y2 = load_data('./data/SCISSORS/PHOTO/*')

    X = X0 + X1 + X2
    y = []
    for i in range(len(X0)):
        y.append([1,0,0])
    for i in range(len(X1)):
        y.append([0,1,0])
    for i in range(len(X2)):
        y.append([0,0,1])

    mirX = []
    miry = []
    for i in range(len(X)):
        mirX.append(cv.flip(X[i].copy(), 1))
        miry.append(y[i])

    X = np.array(X + mirX)
    X_normalized = X.copy()/255
    original_labels = np.array(y + miry)

    predictions = reg_model.predict(X_normalized)
    cropped_images = []
    labels = []
    for i in range(len(X_normalized)):
        x, y, w, h = predictions[i]
        print([x, y, w, h])
        x = int(x * 320)
        y = int(y * 180)
        w = int(w * 320)
        h = int(h * 180)
        print([x, y, w, h])
        image = crop_around_box(X[i],[x, y, w, h] )
        cv.imshow("predicted hand", cv.resize(image.copy(),(320, 320)))
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):
            cropped_images.append(image)
            labels.append(original_labels[i])
            print("saved image")
        elif key == ord('q'):
            print("discarded image")


        np.save('./data/OnlyHands/model_photos', cropped_images)
        np.save('./data/OnlyHands/model_labels', labels)
