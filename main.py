from Network import *

import pygame, sys
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale

def drawCircle( screen, x, y ):
    pygame.draw.circle( screen, (255, 255, 255), ( x, y ), 10 )

pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption('Draw a digit')

font = pygame.font.SysFont("timesnewroman", 15)
text = font.render("Submit", True, (255, 0, 0), (150, 150, 150))
textRect = text.get_rect()
textRect.center = (255, 267)
font1 = pygame.font.SysFont("timesnewroman", 30)

net = Network("weights.txt")

def pix_filter(pixval):
    return 1 if pixval == 255 else 0

def submit():
    # grab pixel array
    pixels = np.array(pygame.surfarray.array_green(screen))

    # rescale, resize, refactor
    pixels = pixels.transpose()
    pixels = np.vectorize(pix_filter)(pixels)
    pixels = rescale(pixels.astype(np.float64), 0.1, anti_aliasing=False)

    # feed to network
    result = net.feed(pixels.flatten()[:, np.newaxis])
    result = np.argmax(result)

    screen.fill((0, 0, 0))
    text1 = font.render(f"the number is {result}", True, (255, 0, 0))
    screen.blit(text1, (100, 0))

isPressed = False
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            isPressed = True
            mouse = pygame.mouse.get_pos()
            if mouse[0] >= 255 and mouse[1] >= 267:
                submit()
        elif event.type == pygame.MOUSEBUTTONUP:
            isPressed = False
        elif event.type == pygame.MOUSEMOTION and isPressed == True:         
            ( x, y ) = pygame.mouse.get_pos()       # returns the position of mouse cursor
            drawCircle( screen, x, y )
    
    screen.blit(text, textRect)
    pygame.display.flip()