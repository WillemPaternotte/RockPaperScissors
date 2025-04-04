import cv2 as cv
import numpy as np
# Example file showing a basic pygame "game loop"
import pygame

# pygame setup
pygame.init()
screen = pygame.display.set_mode((128, 128))
clock = pygame.time.Clock()
running = True
counter, text = 0, ''.rjust(3)
pygame.time.set_timer(pygame.USEREVENT, 1000)
font = pygame.font.SysFont('Consolas', 30)

vid = cv.VideoCapture(0)

start_ticks=pygame.time.get_ticks() #starter tick
frames = []
while (running):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frames.append(frame)
    # cv.imshow("webcam", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            counter = 3
        if e.type == pygame.QUIT:
            running = False

    seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
    if seconds > 1.2 and counter>-1:  # if more than 10 seconds close the game
        text = str(counter).rjust(3) if counter > 0 else 'shoot!'
        counter -= 1
        start_ticks = pygame.time.get_ticks()

    screen.fill((255, 255, 255))
    screen.blit(font.render(text, True, (0, 0, 0)), (32, 48))
    pygame.display.flip()
    clock.tick(60)

for i in range(len(frames)):
    cv.imshow("webcam", frames[i])
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

np.save("./data/video", frames)
pygame.quit()

# After the loop release the cap objectq
vid.release()
# Destroy all the windows
cv.destroyAllWindows()