import pygame
from pygame import Vector2
import numpy as np
import pickle

canvas_size = Vector2(28, 28)
pixel_size = 15

pygame.init()

screen = pygame.display.set_mode(canvas_size*pixel_size)

grid = np.zeros((int(canvas_size.y), int(canvas_size.x)))

model = pickle.load(open("fold0.nn", "rb"))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                grid = np.zeros((int(canvas_size.y), int(canvas_size.x)))

    cursor_pos = Vector2(pygame.mouse.get_pos()) // pixel_size
    mouse_btn_states = pygame.mouse.get_pressed(3)

    if mouse_btn_states[0]:
        grid[int(cursor_pos.x), int(cursor_pos.y)] = 1
    elif mouse_btn_states[2]:
        grid[int(cursor_pos.x), int(cursor_pos.y)] = 0

    screen.fill((0, 0, 0))
    for x in range(int(canvas_size.x)):
        for y in range(int(canvas_size.y)):
            if grid[x, y]:
                pygame.draw.rect(screen,(255, 255, 255), (x*pixel_size, y*pixel_size, pixel_size, pixel_size))

    f = np.matrix(grid.flatten())
    prediction = model.predict(f)
    prediction = np.squeeze(np.asarray(prediction))
    print([f"{v:.2f}" for v in prediction])

    pygame.display.update()