import time

import cv2 as cv
import numpy as np
# Example file showing a basic pygame "game loop"
import pygame
from enum import Enum
import glob
import tensorflow as tf
from model_test import iou_loss, iou_metric
from croparoundbox import crop_around_box

class Hand(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


# pygame setup
pygame.init()
screen = pygame.display.set_mode((750, 400))
clock = pygame.time.Clock()
running = True
counter, text = -10, ''.rjust(3)
pygame.time.set_timer(pygame.USEREVENT, 1000)
font = pygame.font.SysFont('Consolas', 30)

# Start video
vid = cv.VideoCapture(0)

reg_model = tf.keras.models.load_model("model_saves/my_model_alldata_new2.keras", custom_objects={'iou_loss': iou_loss, 'iou_metric': iou_metric})
cls_model = tf.keras.models.load_model("classification_model_saves/my_model_nokernel.keras")

ret, frame = vid.read()
last_frame = cv.resize(frame, (320, 180))

last_key = -1

while running:


    ret, frame = vid.read()
    frame = cv.resize(frame, (320, 180))
    last_frame = frame.copy()

    input_frame = frame.astype(np.float32) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)  # shape (1, 180, 320, 3)

    prediction_box = reg_model.predict(input_frame)[0]
    # print(prediction_box)
    x, y, w, h = prediction_box
    x = int(x*320)
    y = int(y*180)
    w = int(w*320)
    h = int(h*180)
    print([x, y, w, h])
    last_frame = cv.rectangle(last_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    hand_only_frame = crop_around_box(frame, [x,y,w,h])
    input_hand_only_frame = hand_only_frame.astype(np.float32) / 255.0
    input_hand_only_frame = np.expand_dims(input_hand_only_frame, axis=0)  # shape (1, 180, 320, 3)
    prediction_move = cls_model.predict(input_hand_only_frame)[0]
    best_prediction_index = np.argmax(prediction_move)
    # cv.imshow("prediction", last_frame)
    # cv.waitKey(60)


    screen.fill((255, 255, 255))
    hand_only_surface = cv.cvtColor(hand_only_frame, cv.COLOR_BGR2RGB)
    hand_only_surface = pygame.surfarray.make_surface(cv.resize(hand_only_surface, (180,180)))
    hand_only_surface = pygame.transform.rotate(hand_only_surface, -90)
    screen.blit(hand_only_surface, (460, 110))
    surf = cv.cvtColor(last_frame, cv.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(surf)
    surf = pygame.transform.rotate(surf, -90)
    screen.blit(surf, (130, 110))
    screen.blit(font.render(Hand(best_prediction_index).name + str(prediction_move[best_prediction_index]), True, (0, 0, 0)), (20, 80))
    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

pygame.quit()

# After the loop release the cap objectq
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
