import time

import cv2 as cv
import numpy as np
# Example file showing a basic pygame "game loop"
import pygame
from enum import Enum
import glob
import os


def get_last_video_index(name):
    list_of_files = glob.glob('./data/' + name + '/Video/*')  # * means all if need specific format then *.csv
    if len(list_of_files) <= 0:
        return -1
    latest_file = max(list_of_files, key=os.path.getctime)
    last_index = latest_file[20 + len(name):]
    return int(last_index[:-4])



last_file_index = [get_last_video_index('ROCK'), get_last_video_index('PAPER'), get_last_video_index('SCISSORS')]
print(last_file_index)

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

vid = cv.VideoCapture(0)

start_ticks = pygame.time.get_ticks()  # starter tick
frames = []
ret, frame = vid.read()
last_frame = cv.resize(frame, (320, 180))

last_key = -1
while (running):

    # Capture the video frame
    # by frame
    if counter > 0:
        ret, frame = vid.read()
        frame = cv.resize(frame, (320, 180))
        last_frame = frame
        frames.append(frame)

    elif 0 <= last_key <= 2:
        print("save frames")
        last_file_index[last_key] += 1
        np.save("./data/" + Hand(last_key).name + "/Video/video_" + str(last_file_index[last_key]) + ".npy", frames)
        frames = []
        time.sleep(0.1)
        print("reset frames")
        ret, frame = vid.read()
        frame = cv.resize(frame, (320, 180))
        np.save("./data/" + Hand(last_key).name + "/Photo/photo_" + str(last_file_index[last_key]) + ".npy", frame)

        last_key = -1

    # cv.imshow("webcam", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            last_key = e.key - 49
            if 0 <= last_key <= 2:
                counter = 4
                frames = []
                print("reset frames")
        if e.type == pygame.QUIT:
            running = False

    seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
    if seconds > 1.1 and counter > 0:  # if more than 10 seconds close the game
        counter -= 1
        text = str(counter).rjust(3) if counter > 0 else 'shoot!'

        start_ticks = pygame.time.get_ticks()

    screen.fill((255, 255, 255))
    screen.blit(font.render(text, True, (0, 0, 0)), (20, 150))
    screen.blit(font.render("Press a number to record the action", True, (0, 0, 0)), (20, 20))
    screen.blit(font.render("1 for ROCK, 2 for PAPER, 3 for SCISSORS", True, (0, 0, 0)), (20, 40))
    screen.blit(font.render("The game will count down from 3,", True, (0, 0, 0)), (20, 60))
    screen.blit(font.render("do your action at SHOOT! ", True, (0, 0, 0)), (20, 80))
    surf = cv.cvtColor(last_frame, cv.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(surf)
    surf = pygame.transform.rotate(surf, -90)
    screen.blit(surf, (150, 110))
    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

for i in range(len(frames)):
    cv.imshow("webcam", frames[i])
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

pygame.quit()

# After the loop release the cap objectq
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
