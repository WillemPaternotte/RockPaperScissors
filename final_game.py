import random
import time
import cv2 as cv
import numpy as np
import pygame
from enum import Enum
import glob
import os
import tensorflow as tf
from model_test import iou_loss, iou_metric
from croparoundbox import crop_around_box

reg_model = tf.keras.models.load_model("model_saves/my_model_alldata_new2.keras", custom_objects={'iou_loss': iou_loss, 'iou_metric': iou_metric})
cls_model = tf.keras.models.load_model("classification_model_saves/my_model_nokernel.keras")

rock_image = pygame.image.load("./pygame_images/rock.jpg")
paper_image = pygame.image.load("./pygame_images/paper.jpg")
scissors_image = pygame.image.load("./pygame_images/scissors.jpg")

class Hand(Enum):
    DRAW = -1
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

def introduction_text(screen):
    screen.blit(font.render("Welcome to Rock Paper Scissor", True, (0, 0, 0)), (20, 20))
    screen.blit(font.render("Press space to play!", True, (0, 0, 0)), (20, 40))
    screen.blit(font.render("The game will count down from 3,", True, (0, 0, 0)), (20, 60))
    screen.blit(font.render("do your action at SHOOT! ", True, (0, 0, 0)), (20, 80))

def predict_bounding_box(image):
    input_frame = image.astype(np.float32) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)  # shape (1, 180, 320, 3)
    prediction_box = reg_model.predict(input_frame)[0]
    # print(prediction_box)
    x, y, w, h = prediction_box
    x = int(x * 320)
    y = int(y * 180)
    w = int(w * 320)
    h = int(h * 180)

    return x, y, w, h

def predict_hand_action(image):
    input_hand_only_frame = image.astype(np.float32) / 255.0
    input_hand_only_frame = np.expand_dims(input_hand_only_frame, axis=0)  # shape (1, 180, 320, 3)
    prediction_move = cls_model.predict(input_hand_only_frame)[0]
    best_prediction_index = np.argmax(prediction_move)
    return best_prediction_index, prediction_move[best_prediction_index]

def display_image(screen, image, size, location):
    surf = cv.cvtColor(cv.resize(image, size), cv.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(surf)
    surf = pygame.transform.rotate(surf, -90)
    screen.blit(surf, location)

def display_ai_move(screen, move, location):
    if move == 0:
        screen.blit(rock_image, location)
    if move == 1:
        screen.blit(paper_image, location)
    if move == 2:
        screen.blit(scissors_image, location)
def check_win(move_player_a, move_player_b, score):
    if (move_player_a == 0 and move_player_b == 2) or (move_player_a == 1 and move_player_b == 0) or (move_player_a == 2 and move_player_b == 1):
        print("player A wins")
        return (score[0]+1, score[1])
    elif (move_player_a == 0 and move_player_b == 0) or (move_player_a == 1 and move_player_b == 1) or (move_player_a == 2 and move_player_b == 2):
        print("it's a draw")
        return score
    else:
        print("Player B wins")
        return (score[0], score[1]+1)
# pygame setup
pygame.init()
screen = pygame.display.set_mode((1080, 720))
clock = pygame.time.Clock()
running = True
counter, text = -10, ''.rjust(3)
pygame.time.set_timer(pygame.USEREVENT, 1000)
font = pygame.font.SysFont('Consolas', 30)

vid = cv.VideoCapture(0)

start_ticks = pygame.time.get_ticks()  # starter tick

ret, frame = vid.read()
last_frame = cv.resize(frame, (320, 180))
shoot_frame = frame
last_key = -1
ai_move = -1
predicted_hand, hand_prediction_confidence = -1, -1
minimum_confidence = 0.66
score = [0,0]
while (running):

    # Capture the video frame
    ret, frame = vid.read()
    frame = cv.resize(frame, (320, 180))
    last_frame = frame.copy()

    x, y, w, h = predict_bounding_box(last_frame)

    last_frame = cv.rectangle(last_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    hand_only_frame = crop_around_box(frame, [x,y,w,h])




    screen.fill((255, 255, 255))
    screen.blit(font.render(text, True, (0, 0, 0)), (20, 150))
    introduction_text(screen)
    if counter == 0:
        shoot_frame = hand_only_frame
        counter = -1
        ai_move = random.randint(0,2)
        predicted_hand, hand_prediction_confidence = predict_hand_action(shoot_frame)
        if hand_prediction_confidence > minimum_confidence:
            score = check_win(predicted_hand, ai_move, score)
        else:
            print("no hand detected")

    if counter == -1:

        display_image(screen, shoot_frame, (180, 180), (130, 400))
        if hand_prediction_confidence > minimum_confidence:
            screen.blit(font.render(Hand(predicted_hand).name+": " + str(int(hand_prediction_confidence*100))+"%", True, (0, 0, 0)), (20, 370))
            display_ai_move(screen, ai_move, (620, 470))
            screen.blit(
                font.render("AI move: " + Hand(ai_move).name, True,
                            (0, 0, 0)), (620, 370))
        else:
            screen.blit(
                font.render(str(int(hand_prediction_confidence*100))+ "% confidence is to low, couldn't interpret hand", True,
                            (0, 0, 0)), (20, 370))


    display_image(screen, hand_only_frame, (180,180), (460, 110))
    display_image(screen, last_frame, (320, 180), (130,110))

    screen.blit(
        font.render(str(score[0]) + " : " + str(score[1]), True,
                    (0, 0, 0)), (400, 670))

    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                counter = 4
        if e.type == pygame.QUIT:
            running = False

    seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
    if seconds > 1.1 and counter > 0:  # if more than 10 seconds close the game
        counter -= 1
        if counter > 0:
            text = str(counter).rjust(3)
        else:
            text = 'shoot!'


        start_ticks = pygame.time.get_ticks()



pygame.quit()

# After the loop release the cap objectq
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
