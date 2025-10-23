import pygame, sys
from pygame.locals import *

pygame.init()

FPS = 30
fpsClock = pygame.time.Clock()

DISPLAY = pygame.display.set_mode((600, 600), 0, 32)
pygame.display.set_caption('Breaking assistant')


while True:
    pass

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(FPS)
