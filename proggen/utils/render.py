#!/usr/bin/env python
# coding=utf-8

import time
import numpy as np
from matplotlib import colormaps
import pygame
import cv2

def rotate_point(p, angle,):
    x, y = p
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


class PygameRender(object):
    def __init__(
        self, fps, screen_width, screen_height, ppm,
        plot_trajs=True,
        screen_name = 'pygame',
        cmap = 'gist_rainbow', background_color = (255, 255, 255),
    ):
        self.fps = fps
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ppm = ppm
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height),)
        pygame.display.set_caption(screen_name)
        self.clock = pygame.time.Clock()
        self.cmap = colormaps[cmap]
        self.background_color = background_color
        self.plot_trajs = plot_trajs
        self.trajs = []

    def render(self, obs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise ValueError('Quit')
        self.trajs.append(obs)
        obj_names = sorted(list(obs.keys()))
        obj2colors = {obj_name: self.cmap(i / len(obj_names)) for i, obj_name in enumerate(obj_names)}
        self.screen.fill(self.background_color)
        for obj_name, obj in obs.items():
            color = obj2colors[obj_name]
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            if obj.shape == 'circle':
                pygame.draw.circle(
                    self.screen, color,
                    (int(obj.position[0] * self.ppm), self.screen_height - int(obj.position[1] * self.ppm)), int(obj.radius * self.ppm))
            elif obj.shape == 'polygon':
                vertices = [rotate_point(v, obj.angle) for v in obj.vertices]
                vertices = [(v[0] + obj.position[0], v[1] + obj.position[1]) for v in vertices]
                vertices = [(int(v[0] * self.ppm), self.screen_height - int(v[1] * self.ppm)) for v in vertices]
                pygame.draw.polygon(self.screen, color, vertices)
            elif obj.shape == 'edge':
                vertices = [rotate_point(v, obj.angle) for v in obj.vertices]
                vertices = [(v[0] + obj.position[0], v[1] + obj.position[1]) for v in vertices]
                vertices = [(int(v[0] * self.ppm), self.screen_height - int(v[1] * self.ppm)) for v in vertices]
                pygame.draw.lines(self.screen, color, True, vertices)
            else:
                raise ValueError(f'Unknown shape: {obj.shape}')
            # Draw an arrow from the center of the object to the direction of the object
            v = rotate_point((0, min(self.screen_width, self.screen_height) / self.ppm * 0.05,), obj.angle)
            v = (v[0] + obj.position[0], v[1] + obj.position[1])
            pygame.draw.line(
                self.screen, (0, 0, 0),
                (int(obj.position[0] * self.ppm), self.screen_height - int(obj.position[1] * self.ppm)),
                (v[0] * self.ppm, self.screen_height - v[1] * self.ppm),
            )
            if self.plot_trajs:
                # Draw position trajectory of the object as lines
                for i in range(0, len(self.trajs)):
                    if obj_name in self.trajs[i]:
                        pygame.draw.circle(
                            self.screen, (0, 0, 0),
                            (int(self.trajs[i][obj_name].position[0] * self.ppm), self.screen_height - int(self.trajs[i][obj_name].position[1] * self.ppm)), 2)
                        if i > 0 and obj_name in self.trajs[i-1]:
                            pygame.draw.line(
                                self.screen, (0, 0, 0),
                                (int(self.trajs[i-1][obj_name].position[0] * self.ppm), self.screen_height - int(self.trajs[i-1][obj_name].position[1] * self.ppm)),
                                (int(self.trajs[i][obj_name].position[0] * self.ppm), self.screen_height - int(self.trajs[i][obj_name].position[1] * self.ppm)),
                            )
        pygame.display.flip()
        self.clock.tick(self.fps)
        # time.sleep(1./self.fps)

    def get_image(self):
        return pygame.surfarray.array3d(pygame.display.get_surface()).swapaxes(0, 1)

    def close(self):
        pygame.quit()


class OpenCVRender(object):
    def __init__(
        self, screen_width, screen_height, ppm,
        cmap = 'gist_rainbow', background_color = (255, 255, 255),
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ppm = ppm
        pygame.init()
        pygame.display.init()
        self.cmap = colormaps[cmap]
        self.background_color = background_color

    def render(self, obs):
        obj_names = sorted(list(obs.keys()))
        # assert False, {obj_name: self.cmap(i / len(obj_names)) for i, obj_name in enumerate(obj_names)}
        obj2colors = {obj_name: self.cmap(i / len(obj_names)) for i, obj_name in enumerate(obj_names)}
        img = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * self.background_color
        img = img.astype(np.uint8)
        for obj_name, obj in obs.items():
            color = obj2colors[obj_name]
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            if obj.shape == 'circle':
                cv2.circle(
                    img, (int(obj.position[0] * self.ppm), self.screen_height - int(obj.position[1] * self.ppm)),
                    int(obj.radius * self.ppm), color, -1)
            elif obj.shape == 'polygon':
                vertices = [rotate_point(v, obj.angle) for v in obj.vertices]
                vertices = [(v[0] + obj.position[0], v[1] + obj.position[1]) for v in vertices]
                vertices = [(int(v[0] * self.ppm), self.screen_height - int(v[1] * self.ppm)) for v in vertices]
                cv2.fillPoly(img, [np.array(vertices)], color)
            elif obj.shape == 'edge':
                vertices = [rotate_point(v, obj.angle) for v in obj.vertices]
                vertices = [(v[0] + obj.position[0], v[1] + obj.position[1]) for v in vertices]
                vertices = [(int(v[0] * self.ppm), self.screen_height - int(v[1] * self.ppm)) for v in vertices]
                cv2.polylines(img, [np.array(vertices)], True, color)
            else:
                raise ValueError(f'Unknown shape: {obj.shape}')

        for obj_name, obj in obs.items():
            # # Draw a cross at the center of the object
            # HALF_CROSS_SIZE = 3
            # cv2.line(
                # img, (int(obj.position[0] * self.ppm)-HALF_CROSS_SIZE, self.screen_height - int(obj.position[1] * self.ppm)-HALF_CROSS_SIZE),
                # (int(obj.position[0] * self.ppm)+HALF_CROSS_SIZE, self.screen_height - int(obj.position[1] * self.ppm)+HALF_CROSS_SIZE),
                # (0, 0, 0), 1)
            # cv2.line(
                # img, (int(obj.position[0] * self.ppm)-HALF_CROSS_SIZE, self.screen_height - int(obj.position[1] * self.ppm)+HALF_CROSS_SIZE),
                # (int(obj.position[0] * self.ppm)+HALF_CROSS_SIZE, self.screen_height - int(obj.position[1] * self.ppm)-HALF_CROSS_SIZE),
                # (0, 0, 0), 1)

            # Draw an arrow from the center of the object to the direction of the object
            v = rotate_point((0, min(self.screen_width, self.screen_height) / self.ppm * 0.05,), obj.angle)
            v = (v[0] + obj.position[0], v[1] + obj.position[1])
            cv2.arrowedLine(
                img, (int(obj.position[0] * self.ppm), self.screen_height - int(obj.position[1] * self.ppm
                )), (int(v[0] * self.ppm), self.screen_height - int(v[1] * self.ppm)), (0, 0, 0), 1, tipLength=0.2)
            # # Draw the base of the arrow
            # v1 = rotate_point((0, min(self.screen_width, self.screen_height) / self.ppm * 0.01,), obj.angle + np.pi / 2)
            # v2 = rotate_point((0, min(self.screen_width, self.screen_height) / self.ppm * 0.01,), obj.angle - np.pi / 2)
            # v1 = (v1[0] + obj.position[0], v1[1] + obj.position[1])
            # v2 = (v2[0] + obj.position[0], v2[1] + obj.position[1])
            # cv2.line(
                # img, (int(v1[0] * self.ppm), self.screen_height - int(v1[1] * self.ppm)),
                # (int(v2[0] * self.ppm), self.screen_height - int(v2[1] * self.ppm)), (0, 0, 0), 1)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # assert False
        return img

    def close(self):
        pass

