import pygame as pg
import math
import random
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os
import glob
from enum import Enum

pg.init()

pg.font.init()
stats_font = pg.font.SysFont('segoeui', 15)
checkpoint_font = pg.font.SysFont('segoeui', 20)

screen_width = 1300
screen_height = 800
screen = pg.display.set_mode((screen_width, screen_height)) 

def check_rect_collision(rect, objects):
    for i in range(len(objects)):
        if rect.colliderect(objects[i]):
            return i
    return -1

def draw_walls(walls):
    for wall in walls:
        pg.draw.rect(screen, wall.colour, [wall.rect.x, wall.rect.y, wall.rect.width, wall.rect.height])

def draw_finish(tile_rows):
    finish = checkpoints[-1]

    tile_size = int(finish.rect.height / tile_rows)
    tiles_per_row = int(finish.rect.width / tile_size)
    tile_y = finish.y

    for i in range(tile_rows):
        tile_x = finish.x
        for j in range(tiles_per_row):
            if (i + j) % 2 == 0:
                tile_color = pg.color.THECOLORS['black']
            else:
                tile_color = pg.color.THECOLORS['white']

            pg.draw.rect(screen, tile_color, [tile_x, tile_y, tile_size, tile_size])

            tile_x += tile_size
        tile_y += tile_size



def handle_user_car(keys, user_car):
    user_car.handle_user_input(keys)
    user_car.set_current_checkpoint_and_distance(checkpoints)
    user_car.set_and_draw_sonar_arms(nr_arms = 6, arms_scan_range = 180, number_of_points = 9, distance = 50, size = 4, add_back_arm = True)
    user_car.rotate_blit()

def handle_cars(cars):
    for car in cars:
        if not car.is_crashed and not car.is_finished or car.is_best: 
            if not (game_time - game_start_time > min_time_to_reach_first_checkpoint and len(car.checkpoint_times) < 1):
                car.set_and_draw_sonar_arms(nr_arms = 6, arms_scan_range = 180, number_of_points = 9, distance = 50, size = 4, add_back_arm = True)
                car.set_current_checkpoint_and_distance(checkpoints)
                car.rotate_blit()
                car.drive(walls)
            else:
                car.is_idle = True


def reset_car(car):
    car.x = car_spawn_x
    car.y = car_spawn_y
    car.angle = 90


class Sprite():
    def __init__(self, sheet, width, height, end, repeat = False):
        self.sheet = sheet
        self.width = width
        self.height = height
        self.end = end
        self.counterX = 0
        self.counterY = 0
        self.imageWidth = 0
        self.imageHeight = 0
        self.spritesPerRow = 0
        self.spritesPerCol = 0
        self.spriteArea = 0
        self.repeat = repeat

        self.imageWidth = self.sheet.get_rect().size[0]
        self.imageHeight = self.sheet.get_rect().size[1]
        self.spritesPerRow = self.imageWidth / self.width
        self.spritesPerCol = self.imageHeight / self.height

    def update(self):
        self.spriteArea = (self.counterX * self.width, self.counterY * self.height, self.width, self.height)
        self.counterX += 1
        if self.counterX == self.spritesPerRow:
            self.counterX = 0
            self.counterY += 1
        if self.counterY == self.spritesPerCol:
            if self.repeat:
                self.counterX = 0
                self.counterY = 0

class Wall():
    def __init__(self, x, y, width, height, colour = pg.color.THECOLORS["darkgray"]):
        self.surf = pg.Surface((int(width), int(height))) 
        self.rect = self.surf.get_rect()
        self.rect.center = x, y
        self.colour = colour

class Checkpoint():
    def __init__(self, x, y, width, height, nr, focus_off_x, focus_off_y, colour = (110,125,110)):
        self.x = x
        self.y = y
        self.surf = pg.Surface((int(width), int(height)))
        self.rect = self.surf.get_rect()
        self.rect.center = x + int(width / 2), y + int(height / 2)
        self.nr = nr
        self.colour = colour
        self.focus_point = (self.x + focus_off_x, self.y + focus_off_y)

class Arm():
    def __init__(self, points, angle):
        self.points = points
        self.angle_from_car = angle
        self.arm_length = len(self.points)

class NeuralNetwork(object):
    def __init__(self):
        self.nr_of_inputs = 6
        self.nr_of_outputs = 4
        self.weights1 = np.random.rand(self.nr_of_inputs,20)
        self.weights2 = np.random.rand(20,self.nr_of_outputs)

    def feedforward(self, x):
        self.layer1 = relu(np.dot(x, self.weights1))
        self.output = softmax(np.dot(self.layer1, self.weights2))
        return np.argmax((self.output[0]))

class CarType(Enum):
    CROSSOVER = 1
    MUTATION = 2
    BEST = 3
    SAVED = 4
    RANDOM = 5
    USER = 6

class Levels(Enum):
    BONOBORACING123 = 1
    TURNAROUNDANDAROUND = 2

class Car():
    def __init__(self, car_img):
        self.x = car_spawn_x
        self.y = car_spawn_y
        self.angle = 90
        self.vel = 0
        self.max_speed = 19
        self.brake_speed = -1
        self.drive_speed = 1
        self.id = id(self)
        self.car_type = CarType.CROSSOVER
        
        self.image = pg.image.load("jaap_game/images/cars/"+car_img+".png")
        self.rect = self.image.get_rect()
        self.rect.center = self.x, self.y
        self.arms = []
        self.visible = True
        self.show_collision_rect = False
        self.can_drive = True
        self.is_crashed = False
        self.is_idle = False
        self.is_finished = False
        self.is_best = False
        self.colour = pg.color.THECOLORS["black"]
        self.explosion_sheet = Sprite(pg.image.load("jaap_game/images/explosion.png"), 200, 240, 0, False)
        self.fire_sheet = Sprite(pg.image.load("jaap_game/images/fire.png"), 20, 20, 0, True)

        self.current_checkpoint_nr = 0
        self.nr_of_wrong_checkpoints = 0
        self.dist_to_next_checkpoint = -1
        self.laps_finished = 0
        self.finish_time = 0
        self.checkpoint_times = []

        self.fitness = 0
        self.prob = 0
        self.cum_prob = 0

        self.brain = NeuralNetwork()

    def drive(self, walls):
        self.rect.center = self.x, self.y
        if check_rect_collision(self.rect, walls) != -1 or not self.can_drive:
            self.is_crashed = True
            return
        self.n_drive()
        
    def n_drive(self):
        arm_lengths = []
        for arm in self.arms:
            arm_lengths.append(arm.arm_length)
        direction = self.brain.feedforward([arm_lengths])
        if direction == 0:
            self.drive_forward()
        if direction == 1:
            self.brake()
        if direction == 2:
            self.angle += -6
        if direction == 3:
            self.angle += 6
        if direction == 4:
            pass
        self.set_new_position()

    def handle_user_input(self, keys):
        if check_rect_collision(self.rect, walls) == -1:
            if keys[pg.K_LEFT]:
                self.angle += 6
            if keys[pg.K_RIGHT]:
                self.angle += -6
            if keys[pg.K_UP]:
                self.drive_forward()
            if keys[pg.K_DOWN]:
                self.brake()

    def drive_forward(self):
        if self.vel + self.drive_speed <= self.max_speed:
            self.increase_speed(self.drive_speed)

    def brake(self):
        if abs(self.vel + self.brake_speed) <= self.max_speed:
            self.increase_speed(self.brake_speed)

    def increase_speed(self, amount):
        self.vel += amount

    def set_new_position(self):
        self.x += int(self.vel*math.sin(math.radians(self.angle)))
        self.y += int(self.vel*math.cos(math.radians(self.angle)))

    def rotate_blit(self):
        if self.angle > 360:
            self.angle = 0
        elif self.angle < 0:
            self.angle = 360

        rotated_image = pg.transform.rotate(self.image, self.angle)
        rotRect = rotated_image.get_rect()
        rotRect.center = self.x, self.y
        
        if self.show_collision_rect:
            self.image.set_colorkey(self.colour)
        
        if self.visible:
            screen.blit(rotated_image, rotRect)

    def set_and_draw_sonar_arms(self, nr_arms, arms_scan_range, number_of_points, distance, size, add_back_arm = False):
        self.arms = []
        self.brain.nr_of_inputs = nr_arms
        arms_spread = 0
        if nr_arms > 1:
            arms_spread = arms_scan_range / (nr_arms - 1)
        for i in range(nr_arms):
            arm_points = []
            point_x = self.x
            point_y = self.y

            if add_back_arm:
                angle = self.angle + 180
                add_back_arm = not add_back_arm
                arms_spread = arms_scan_range / nr_arms
            else:
                angle = ((self.angle) + arms_scan_range / 2) - arms_spread * i

            for j in range(1, number_of_points):
                point_x += int(distance*math.sin(math.radians(angle)))
                point_y += int(distance*math.cos(math.radians(angle)))

                point_surf = pg.Surface((size, size))
                point_rect = point_surf.get_rect()
                point_rect.center = point_x, point_y
                if check_rect_collision(point_rect, walls) != -1:
                    break

                arm_points.append((point_x, point_y))

                if not hide_car_arms:
                    screen.blit(point_surf, point_rect)

            self.arms.append(Arm(arm_points, angle - self.angle))

    def set_current_checkpoint_and_distance(self, checkpoints):
        self.rect.center = self.x, self.y

        for checkpoint in checkpoints:
            next_checkpoint = checkpoint
            if check_rect_collision(self.rect, [checkpoint.rect]) != -1:
                if self.current_checkpoint_nr + 1 == checkpoint.nr:
                    self.current_checkpoint_nr = checkpoint.nr
                    self.checkpoint_times.append([checkpoint.nr, game_time - game_start_time])
                    if self.current_checkpoint_nr == len(checkpoints):
                        self.finish_time = game_time - game_start_time
                        next_checkpoint = checkpoints[1]
                        self.laps_finished += 1
                        self.can_drive = False
                        self.is_finished = True
                # elif self.current_checkpoint_nr + 1 != checkpoint.nr:
                #     self.nr_of_wrong_checkpoints += 1

        if self.current_checkpoint_nr + 1 < len(checkpoints):
            next_checkpoint = checkpoints[self.current_checkpoint_nr]
        self.dist_to_next_checkpoint = math.hypot(self.x - next_checkpoint.focus_point[0], self.y - next_checkpoint.focus_point[1])


#Level specific
level_to_load = Levels.BONOBORACING123
car_spawn_x = 0
car_spawn_y = 0
new_gen_time = 0
min_time_to_reach_first_checkpoint = 0
checkpoints = []
walls = []
pop_size = 300

#General
delay = 20
cars = get_initial_pop()
weights_directory = 'jaap_game/weights/'+str(time.time())+'/'

#Time related
total_gens = 1
ticks = 1
game_time = 0
game_start_time = 0

#Visibility
hide_best_cars = False
hide_mutation_cars = False
hide_crossover_cars = False
hide_random_cars = False
hide_car_arms = True

#Other cars
user_car = Car("car4")
user_car.car_type = CarType.USER
saved_ai_car = Car("car8")
saved_ai_car.brain = get_saved_car_brain()
saved_ai_car.car_type = CarType.SAVED

#Plot related
fastest_finish_times = []
best_fitness = []

if level_to_load == Levels.BONOBORACING123:
    checkpoints = [
    Checkpoint(x = 690, y = 40, width = 50, height = 160, focus_off_x = 10, focus_off_y = 85, nr = 1),
    Checkpoint(x = 950, y = 40, width = 50, height = 160, focus_off_x = 10, focus_off_y = 140, nr = 2),
    Checkpoint(x = 1055, y = 575, width = 200, height = 50, focus_off_x = 10, focus_off_y = 10, nr = 3),
    Checkpoint(x = 327, y = 600, width = 50, height = 85, focus_off_x = 10, focus_off_y = 10, nr = 4),
    Checkpoint(x = 45, y = 210, width = 200, height = 40, focus_off_x = 120, focus_off_y = 10, nr = 5)]
    walls = [
    Wall(screen_width/2, 0, screen_width, 90),
    Wall(screen_width/2, screen_height, screen_width, 90),
    Wall(0, screen_height/2, 90, screen_height),
    Wall(screen_width, screen_height/2, 90, screen_height),

    Wall(580, 65, 80, 100),
    Wall(860, 180, 80, 100),
    Wall(screen_width/2, screen_height/2, screen_width/1.6, screen_height/2),
    Wall(850, 620, 410, 95),
    #Wall(1150, 500, 50, 200),
    Wall(200, 750, 600, 130),
    Wall(270, 570, 250, 60),
    Wall(20, 400, 250, 60),
    Wall(300, 270, 250, 60)]
    car_spawn_x = 300
    car_spawn_y = 120
    new_gen_time = 8
    min_time_to_reach_first_checkpoint = 1.1

if level_to_load == Levels.TURNAROUNDANDAROUND:
    checkpoints = [
    Checkpoint(x = 260, y = 140, width = 110, height = 20, focus_off_x = 85, focus_off_y = 10, nr = 1),
    Checkpoint(x = 400, y = 40, width = 20, height = 100, focus_off_x = 10, focus_off_y = 80, nr = 2),
    Checkpoint(x = 490, y = 160, width = 110, height = 20, focus_off_x = 60, focus_off_y = 10, nr = 3),
    Checkpoint(x = 490, y = 300, width = 110, height = 20, focus_off_x = 30, focus_off_y = 20, nr = 4)]

    walls = [
    Wall(screen_width/2, 0, screen_width, 90),
    Wall(screen_width/2, screen_height, screen_width, 90),
    Wall(0, screen_height/2, 90, screen_height),
    Wall(screen_width, screen_height/2, 90, screen_height),

    Wall(140, 150, 230, 20),
    Wall(180, 250, 440, 20),
    Wall(430, 300, 120, 320),
    Wall(250, 85, 20, 150),
    Wall(610, 150, 20, 350)]

    car_spawn_x = 70
    car_spawn_y = 200
    new_gen_time = 5
    min_time_to_reach_first_checkpoint = 0.7

run = True
while run:
    pg.time.delay(delay)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

    ticks += 1
    game_time = ticks * (delay / 1000)

    screen.fill((85,96,91))

    draw_walls(walls)
    draw_checkpoints(checkpoints)
    draw_finish(tile_rows = 2)

    handle_cars(cars)
    handle_cars([saved_ai_car])
    handle_user_car(pg.key.get_pressed(), user_car)

    ga = GA(cars)
    if (game_time - game_start_time) > new_gen_time or ga.all_have_crashed: #or ga.fastest_car.is_finished or ga.fastest_car.is_crashed:
        total_gens += 1

        game_start_time = game_time
        
        new_pop = ga.get_new_gen()
        cars = new_pop

        if ga.fastest_car.finish_time > 0:
            fastest_finish_times.append(ga.fastest_car.finish_time)
        else:
            fastest_finish_times.append(np.nan)

        best_fitness.append(ga.fastest_car.fitness)

        save_fastest_car_to_file(ga)

        user_car = Car("car4")
        saved_ai_car = Car("car8")
        saved_ai_car.brain = get_saved_car_brain()

    text = "Gen: "+str(total_gens)
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,45))
    text = "Finished: "+str(ga.number_finished)+"/"+str(pop_size)
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,58))
    text = "Crashed: "+str(ga.number_crashed)+"/"+str(pop_size) 
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,71))
    text = "Idles deleted: " + str(ga.number_idle)+"/"+str(pop_size)
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,84))
    text = "Fastest car: " + str(ga.fastest_car.id)
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,97))
    text = "New round in: " + str(round(game_time - game_start_time, 1)) + " : " + str(new_gen_time)
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,110))
    text = "Fastest: " + str(ga.fastest_car.finish_time)
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,123))

    text = "Hotkeys: Hide (R)andoms, (C)rossovers, (B)est, (M)utations - (S)tats"
    screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["black"]),(55,750))

    pg.display.flip()
pg.quit()