import pygame, os
from pygame.locals import *
import numpy as np
import tensorflow as tf


class Recogniser():
    def __init__(self) -> None:
        self.values = np.zeros((28, 28))

    def draw_lines(self, screen):
        for i in range(28):
            pygame.draw.line(screen, (0, 0, 0), (0, i * 20), (560, i * 20))
            pygame.draw.line(screen, (0, 0, 0), (i * 20, 0), (i * 20, 560))

    def draw_digit(self, screen):
        for i in range(28):
            for j in range(28):
                rect = pygame.Rect(i * 20, j * 20, 20, 20)
                pygame.draw.rect(screen, [(1 - self.values[i][j]) * 255] * 3, rect)

    def display(self, screen):
        self.draw_digit(screen)
        self.draw_lines(screen)

    def draw(self):
        pos = pygame.mouse.get_pos()
        x, y = pos[0] // 20, pos[1] // 20
        if pygame.mouse.get_pressed()[0] and self.values[x][y] != 1:
            self.values[x][y] = 1
            if 0 < x < 28 and 0 < y < 28:
                self.values[x-1][y] = min(1, self.values[x-1][y] + 0.2)
                self.values[x+1][y] = min(1, self.values[x+1][y] + 0.2)
                self.values[x][y-1] = min(1, self.values[x][y-1] + 0.2)
                self.values[x][y+1] = min(1, self.values[x][y+1] + 0.2)
        elif pygame.mouse.get_pressed()[2] and self.values[x][y] > 0:
            self.values[x][y] = 0
            if 0 < x < 28 and 0 < y < 28:
                self.values[x-1][y] = max(0, self.values[x-1][y] - 0.2)
                self.values[x+1][y] = max(0, self.values[x+1][y] - 0.2)
                self.values[x][y-1] = max(0, self.values[x][y-1] - 0.2)
                self.values[x][y+1] = max(0, self.values[x][y+1] - 0.2)

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return True
                if event.key == K_SPACE:
                    self.predict()
                if event.key == K_c:
                    self.values = np.zeros((28, 28))

    def display_screen(self, screen):
        screen.fill((255, 255, 255))

        self.display(screen)

        pygame.display.update()
        pygame.display.flip()

    def run_logic(self):
        self.draw()

    def predict(self):
        model = tf.keras.models.load_model('mnist.model')
        pred = model.predict(np.expand_dims(self.values, axis=0))
        print(np.argmax(pred))


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Arcade Machine")

    os.environ['SDL_VIDEO_CENTERED'] = "True"

    width, height = 560, 560

    screen = pygame.display.set_mode((width, height))

    done = False
    clock = pygame.time.Clock()
    recogniser = Recogniser()

    while not done:
        done = recogniser.events()
        recogniser.run_logic()
        recogniser.display_screen(screen)

        clock.tick(60)


    