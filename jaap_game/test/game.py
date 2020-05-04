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
from enums import *
from helpers import *

screen_width = 1300
screen_height = 800
screen = pg.display.set_mode((screen_width, screen_height)) 


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

class Level1(object):
    def __init__(self):
        self.checkpoints = [
        Checkpoint(x = 690, y = 40, width = 50, height = 160, focus_off_x = 10, focus_off_y = 85, nr = 1),
        Checkpoint(x = 950, y = 40, width = 50, height = 160, focus_off_x = 10, focus_off_y = 140, nr = 2),
        Checkpoint(x = 1055, y = 575, width = 200, height = 50, focus_off_x = 10, focus_off_y = 10, nr = 3),
        Checkpoint(x = 327, y = 600, width = 50, height = 85, focus_off_x = 10, focus_off_y = 10, nr = 4),
        Checkpoint(x = 45, y = 210, width = 200, height = 40, focus_off_x = 120, focus_off_y = 10, nr = 5)
        ]
        self.walls = [
        Wall(1300/2, 0, 1300, 90),
        Wall(1300/2, 800, 1300, 90),
        Wall(0, 800/2, 90, 800),
        Wall(1300, 800/2, 90, 800),

        Wall(580, 45, 80, 100),
        Wall(860, 200, 80, 100),
        Wall(1300/2, 800/2, 1300/1.6, 800/2),
        Wall(850, 620, 410, 95),
        #Wall(1150, 500, 50, 200),
        Wall(200, 750, 600, 130),
        Wall(270, 570, 250, 60),
        Wall(20, 400, 250, 60),
        Wall(300, 270, 250, 60)]

        self.new_round_time = 8
        self.min_time_to_reach_first_checkpoint = 1.1

class Level2(object):
    def __init__(self):
        self.checkpoints = [
        Checkpoint(x = 260, y = 140, width = 110, height = 20, focus_off_x = 85, focus_off_y = 10, nr = 1),
        Checkpoint(x = 400, y = 40, width = 20, height = 100, focus_off_x = 10, focus_off_y = 80, nr = 2),
        Checkpoint(x = 490, y = 160, width = 110, height = 20, focus_off_x = 60, focus_off_y = 10, nr = 3),
        Checkpoint(x = 490, y = 300, width = 110, height = 20, focus_off_x = 30, focus_off_y = 20, nr = 4)]

        self.walls = [
        Wall(1300/2, 0, 1300, 90),
        Wall(1300/2, 800, 1300, 90),
        Wall(0, 800/2, 90, 800),
        Wall(1300, 800/2, 90, 800),

        Wall(140, 150, 230, 20),
        Wall(180, 250, 440, 20),
        Wall(430, 300, 120, 320),
        Wall(250, 85, 20, 150),
        Wall(610, 150, 20, 350)]

        self.new_round_time = 5
        self.min_time_to_reach_first_checkpoint = 0.7

class Car(object):
    def __init__(self, car_img, car_spawn_x, car_spawn_y):
        self.x = car_spawn_x
        self.y = car_spawn_y
        self.angle = 90
        self.vel = 0
        self.max_speed = 19
        self.brake_speed = -1
        self.drive_speed = 1
        self.id = id(self)
        self.car_type = CarType.CROSSOVER
        self.hide_car_arms = False
        
        self.image = pg.image.load("images/cars/"+car_img+".png")
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
        self.explosion_sheet = Sprite(pg.image.load("images/explosion.png"), 200, 240, 0, False)
        self.fire_sheet = Sprite(pg.image.load("images/fire.png"), 20, 20, 0, True)

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
    
    def handle_user_car(self, keys, user_car):
        user_car.handle_user_input(keys)
        user_car.set_current_checkpoint_and_distance(checkpoints)
        user_car.set_and_draw_sonar_arms(nr_arms = 5, arms_scan_range = 180, number_of_points = 9, distance = 50, size = 4, add_back_arm = False)
        user_car.rotate_blit()

    
    def handle_user_input(self, keys):
        if check_rect_collision(self.rect, walls) == -1:
            if keys[pg.K_LEFT]:
                self.drive_left()
            if keys[pg.K_RIGHT]:
                self.drive_right()
            if keys[pg.K_UP]:
                self.drive_forward()
            if keys[pg.K_DOWN]:
                self.brake()
            if keys[pg.K_s]:
                show_stats_plot()
            if keys[pg.K_b]:
                global hide_best_cars
                hide_best_cars = not hide_best_cars
            if keys[pg.K_m]:
                global hide_mutation_cars
                hide_mutation_cars = not hide_mutation_cars
            if keys[pg.K_c]:
                global hide_crossover_cars
                hide_crossover_cars = not hide_crossover_cars
            if keys[pg.K_r]:
                global hide_random_cars
                hide_random_cars = not hide_random_cars         
            else:
                self.glide()

            self.set_new_position()
        else:
            self.explosion_sheet.update()
            screen.blit(self.explosion_sheet.sheet, (self.x - self.explosion_sheet.width/2, self.y - self.explosion_sheet.height), self.explosion_sheet.spriteArea)

    def drive_left(self):
        self.angle += 10
    
    def drive_right(self):
        self.angle += -10

    def drive_forward(self):
        if self.vel + self.drive_speed <= self.max_speed:
            self.increase_speed(self.drive_speed)

    def brake(self):
        if abs(self.vel + self.brake_speed) <= self.max_speed:
            self.increase_speed(self.brake_speed)
    
    def glide(self):
        if self.vel + self.brake_speed / 8 >= 0:
            self.increase_speed(self.brake_speed / 8)

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

    def set_and_draw_sonar_arms(self, nr_arms, arms_scan_range, number_of_points, distance, size, walls, add_back_arm = False):
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

                if not self.hide_car_arms:
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

class Game(object):

    def __init__(self):
        pg.init()
        pg.font.init()
        self.total_gens = 1
        self.ticks = 1
        self.game_time = 0
        self.game_start_time = 0
        self.level = Level1()
        self.new_round_time = 0
        self.min_time_to_reach_first_checkpoint = 0
        self.game_type = GameType.RL
        self.delay = 20
        self.hide_car_arms = True
        self.user_car = Car("car4", 250, 100)
        self.user_car.car_type = CarType.USER
        self.stats_font = pg.font.SysFont('segoeui', 15)
        self.checkpoint_font = pg.font.SysFont('segoeui', 20)
        
    
    def init_game(self,game_type, level_to_load, car_spawn_x, car_spawn_y):
        self.level_to_load = level_to_load
        self.car_spawn_x = car_spawn_x
        self.car_spawn_y = car_spawn_y

    def draw_walls(self,walls):
        for wall in walls:
            pg.draw.rect(screen, wall.colour, [wall.rect.x, wall.rect.y, wall.rect.width, wall.rect.height])

    def draw_checkpoints(self,checkpoints):
        for checkpoint in checkpoints:
            pg.draw.rect(screen, checkpoint.colour, [checkpoint.x, checkpoint.y, checkpoint.rect.width, checkpoint.rect.height])
            pg.draw.rect(screen, (255,255,50), [checkpoint.focus_point[0], checkpoint.focus_point[1], 6, 6])
            screen.blit(self.checkpoint_font.render(str(checkpoint.nr), 1, (0,0,0)),(checkpoint.focus_point[0] - 10,checkpoint.focus_point[1] - 20))

    def draw_finish(self,tile_rows):
        self.finish = self.level.checkpoints[-1]

        self.tile_size = int(self.finish.rect.height / tile_rows)
        self.tiles_per_row = int(self.finish.rect.width / self.tile_size)
        self.tile_y = self.finish.y

        for i in range(tile_rows):
            self.tile_x = self.finish.x
            for j in range(self.tiles_per_row):
                if (i + j) % 2 == 0:
                    self.tile_color = pg.color.THECOLORS['black']
                else:
                    self.tile_color = pg.color.THECOLORS['white']

                pg.draw.rect(screen, self.tile_color, [self.tile_x, self.tile_y, self.tile_size, self.tile_size])

                self.tile_x += self.tile_size
            self.tile_y += self.tile_size
    
    

       

class Genectic_game(Game):
    def __init__(self):
        self.pop_size = 200
        self.cars = get_initial_pop()
        self.weights_directory = 'jaap_game/weights/'+str(time.time())+'/'
        self.hide_best_cars = False
        self.hide_mutation_cars = False
        self.hide_crossover_cars = False
        self.hide_random_cars = False
        self.fastest_finish_times = []
        self.best_fitness = []

    def get_initial_pop():
        cars = []
        for i in range(pop_size):
            cars.append(Car("car2"))
        return cars
    
    def handle_ga_cars(cars):
        for car in cars:
            if not car.is_crashed and not car.is_finished or car.is_best: 
                if not (game_time - game_start_time > min_time_to_reach_first_checkpoint and len(car.checkpoint_times) < 1):
                    car.set_and_draw_sonar_arms(nr_arms = 5, arms_scan_range = 180, number_of_points = 9, distance = 50, size = 4, add_back_arm = False, walls = self.level.walls)
                    car.set_current_checkpoint_and_distance(checkpoints)
                    car.rotate_blit()
                    car.drive(walls)
                else:
                    car.is_idle = True
    
    def run(self):
        run = True
        while run:
            pg.time.delay(delay)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                    
            screen.fill((85,96,91))

            draw_walls(walls)
            draw_checkpoints(checkpoints)
            draw_finish(tile_rows = 2)

            ticks += 1
            game_time = ticks * (delay / 1000)
            
            handle_ga_cars(cars)
            #handle_cars([saved_ai_car])
            handle_user_car(pg.key.get_pressed(), user_car)

            ga = GA(cars)
            if (game_time - game_start_time) > new_round_time or ga.all_have_crashed and game_type == GameType.GA: #or ga.fastest_car.is_finished or ga.fastest_car.is_crashed:
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
            text = "New round in: " + str(round(new_round_time - (game_time - game_start_time), 1))
            screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,110))
            text = "Fastest: " + str(ga.fastest_car.finish_time)
            screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,123))

            text = "Hotkeys: Hide (R)andoms, (C)rossovers, (B)est, (M)utations - (S)tats"
            screen.blit(stats_font.render(text, 1, pg.color.THECOLORS["black"]),(55,750))

            pg.display.flip()
        pg.quit()

class ReinforcementGame(Game):
    def __init__(self):
        Game.__init__(self)
        self.car = ReinforcementCar("car8", 250, 100)
    
    def step(self,action):
        pg.time.delay(self.delay)

        self.ticks += 1
        self.game_time = self.ticks * (self.delay / 1000)

        screen.fill((85,96,91))

        self.draw_walls(self.level.walls)
        self.draw_checkpoints(self.level.checkpoints)
        self.draw_finish(tile_rows = 2)

        if self.car.is_crashed or (self.game_time - self.game_start_time) >= self.new_round_time:
            self.game_start_time = self.game_time
            self.car = ReinforcementCar("car8", 250, 100)

        #self.handle_car(action, self.car)
        self.car.handle(action, self.level.walls, self.level.checkpoints)
        self.text = "Gametime: " + str(round(self.game_time, 1))
        screen.blit(self.stats_font.render(self.text, 1, pg.color.THECOLORS["white"]),(55,70))
        text = "Next round in: " + str(round(self.new_round_time - (self.game_time - self.game_start_time), 1))
        screen.blit(self.stats_font.render(self.text, 1, pg.color.THECOLORS["white"]),(55,83))

        pg.display.flip()



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



class NeuralNetwork(object):
    def __init__(self):
        self.nr_of_inputs = 5
        self.nr_of_outputs = 5
        self.weights1 = np.random.rand(self.nr_of_inputs,20)
        self.weights2 = np.random.rand(20,self.nr_of_outputs)

    def feedforward(self, x):
        self.layer1 = relu(np.dot(x, self.weights1))
        self.output = softmax(np.dot(self.layer1, self.weights2))
        return np.argmax((self.output[0]))




class GenecticCar(Car):

    def set_fitness(self):
        total_fitness = 0

        for i in range(len(self.checkpoint_times)):
            if i == 0:
                total_fitness += 0.2
            else:
                total_fitness += 1 * (1/(self.checkpoint_times[i][1] - self.checkpoint_times[i - 1][1]))
        if self.dist_to_next_checkpoint > 2:
            total_fitness += (1/self.dist_to_next_checkpoint)

        if self.nr_of_wrong_checkpoints > 0:
            total_fitness = 0
        self.fitness = total_fitness
        
    def drive(self):
        self.rect.center = self.x, self.y
        if check_rect_collision(self.rect, walls) != -1 or not self.can_drive:
            self.is_crashed = True
            return
        arm_lengths = []
        for arm in self.arms:
            arm_lengths.append(arm.arm_length)
        direction = self.brain.feedforward([arm_lengths])
        if direction == 0:
            self.drive_forward()
        if direction == 1:
            self.brake()
        if direction == 2:
            self.drive_right()
        if direction == 3:
            self.drive_left()
        if direction == 4:
            self.glide()
        self.set_new_position()
    
    def reset_car(self):
        self.x = self.car_spawn_x
        self.y = self.car_spawn_y
        self.angle = 90

class ReinforcementCar(Car):

    def rl_drive(self, action, walls):
        self.rect.center = self.x, self.y
        if check_rect_collision(self.rect, walls) != -1 or not self.can_drive:
            self.is_crashed = True
            return
        if action == "forward":
            self.drive_forward()
        if action == "brake":
            self.brake()
        if action == "right":
            self.drive_right()
        if action == "left":
            self.drive_left()
        # if action == "nothing":
        #    self.glide()
        self.set_new_position()
    
    def handle(self, action, walls, checkpoints):
        self.rl_drive(action,walls)
        self.set_current_checkpoint_and_distance(checkpoints)
        self.set_and_draw_sonar_arms(nr_arms = 5, arms_scan_range = 180, number_of_points = 9, distance = 50, size = 4, add_back_arm = False, walls = walls)
        self.rotate_blit()


class GA():
    def __init__(self, cars):
        self.cars = cars
        self.new_pop = []
        self.total_fitness = 0
        self.number_finished = 0
        self.number_crashed = 0
        self.number_idle = 0
        self.all_have_crashed = False
        self.mutation_rate = 0.05
        self.fastest_car = self.cars[0]
        
        for car in self.cars:
            if not car.can_drive:
                self.number_finished += 1
            if car.is_crashed:
                self.number_crashed += 1
            if car.is_idle:
                self.number_idle += 1
            if car.fitness > self.fastest_car.fitness: #car.finish_time < self.fastest_car.finish_time and car.finish_time > 1 or self.fastest_car.finish_time == 0:
                self.fastest_car = car

            car.visible = True
            if hide_best_cars and car.car_type == CarType.BEST:
                car.visible = False
            if hide_random_cars and car.car_type == CarType.RANDOM:
                car.visible = False
            if hide_mutation_cars and car.car_type == CarType.MUTATION:
                car.visible = False
            if hide_crossover_cars and car.car_type == CarType.CROSSOVER:
                car.visible = False

        self.all_have_crashed = self.number_crashed + self.number_idle >= pop_size
          
    def set_total_fitness(self):
        total_fit = 0
        for i in range(pop_size):
            self.cars[i].set_fitness()
            total_fit += self.cars[i].fitness
        self.total_fitness = total_fit

    def set_cars_prob(self):
        for i in range(pop_size):
            self.cars[i].prob = (1 / self.total_fitness) * self.cars[i].fitness

    def set_cum_prob(self):
        cum_prob = 0
        for i in range(pop_size):
            cum_prob += self.cars[i].prob
            self.cars[i].cum_prob = cum_prob

    def select_parent(self): 
        rand = random.random()
        for i in range(pop_size):
            if rand < self.cars[i].cum_prob:
                return self.cars[i]

    def matrix_to_array(self):
        for car in self.cars:
            car.brain.weights1 = car.brain.weights1.flatten()
            car.brain.weights2 = car.brain.weights2.flatten()

    def cut_w1(self,parent1, parent2):
        indicator = 0
        cutted_array = []
        while True:
            size_cut = random.randrange(0,10)
            if indicator + size_cut > (parent1.brain.nr_of_inputs * 20):
                size_cut = (parent1.brain.nr_of_inputs * 20) - indicator
                if indicator % 2 == 0:
                    cutted_array.append(parent1.brain.weights1[indicator:indicator + size_cut])
                else:
                    cutted_array.append(parent2.brain.weights1[indicator:indicator + size_cut])
                return cutted_array
            if indicator % 2 == 0:
                cutted_array.append(parent1.brain.weights1[indicator:indicator + size_cut])
            else:
                cutted_array.append(parent2.brain.weights1[indicator:indicator + size_cut])
            indicator += size_cut
    
    def cut_w2(self,parent1,parent2):
        indicator = 0
        cutted_array = []
        while True:
            size_cut = random.randrange(0,10)
            if indicator + size_cut > (20 * parent1.brain.nr_of_outputs):
                size_cut = (20 * parent1.brain.nr_of_outputs) - indicator
                if indicator % 2 == 0:
                    cutted_array.append(parent1.brain.weights2[indicator:indicator + size_cut])
                else:
                    cutted_array.append(parent2.brain.weights2[indicator:indicator + size_cut])
                return cutted_array
            if indicator % 2 == 0:
                cutted_array.append(parent1.brain.weights2[indicator:indicator + size_cut])
            else:
                cutted_array.append(parent2.brain.weights2[indicator:indicator + size_cut])
            indicator += size_cut
        
    def crossover(self, amount):
        self.matrix_to_array()

        for i in range(amount):
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            child = Car("car2")

            cutted_array_w1 = self.cut_w1(parent1,parent2)
            cutted_array_w2 = self.cut_w2(parent1,parent2)
            
            w1 = np.array([])
            for i in cutted_array_w1:
                w1 = np.concatenate((w1,i),axis=None)
            
            w2 = np.array([])
            for i in cutted_array_w2:
                w2 = np.concatenate((w2,i),axis=None)

            inputs = child.brain.nr_of_inputs
            outputs = child.brain.nr_of_outputs

            child.brain.weights1 = w1.reshape((inputs,20))
            child.brain.weights2 = w2.reshape((20,outputs))
            child.color = (1,1,255)

            child.car_type = CarType.CROSSOVER

            self.new_pop.append(child)
            
    def mutate(self, car):
        new_car = Car("car3")

        new_car.brain.weights1 = new_car.brain.weights1.flatten()
        new_car.brain.weights2 = new_car.brain.weights2.flatten()

        fast_car_brain_weights1 = car.brain.weights1.flatten()
        fast_car_brain_weights2 = car.brain.weights2.flatten()
        
        reached_finish = self.fastest_car.is_finished
        for i in range(len(fast_car_brain_weights1)):
            mutate = random.random() < self.mutation_rate
            if mutate and reached_finish:
                new_car.brain.weights1[i] += random.uniform(fast_car_brain_weights1[i] * 0.95, fast_car_brain_weights1[i] * 1.06)
            elif mutate and not reached_finish:
                new_car.brain.weights1[i] = random.random()
            else:
                new_car.brain.weights1[i] = fast_car_brain_weights1[i]

        for i in range(len(fast_car_brain_weights2)):
            mutate = random.random() < self.mutation_rate
            if mutate and reached_finish:
                new_car.brain.weights2[i] = random.uniform(fast_car_brain_weights2[i] * 0.95, fast_car_brain_weights2[i] * 1.06)
            elif mutate and not reached_finish:
                new_car.brain.weights2[i] = random.randrange(0,10)
            else:
                new_car.brain.weights2[i] = fast_car_brain_weights2[i]
        
        inputs = new_car.brain.nr_of_inputs
        outputs = new_car.brain.nr_of_outputs

        new_car.brain.weights1 = new_car.brain.weights1.reshape((inputs,20))
        new_car.brain.weights2 = new_car.brain.weights2.reshape((20,outputs))

        new_car.car_type = CarType.MUTATION

        return new_car

    def best_to_new_pop(self, amount):
        best_cars = cars[:amount]
        for car in best_cars:
            new_car = Car("car6")
            new_car.brain.weights1 = car.brain.weights1
            new_car.brain.weights2 = car.brain.weights2
            new_car.is_best = True
            new_car.car_type = CarType.BEST
            new_car.id = car.id
            self.new_pop.append(new_car)
    
    def fill_new_pop_wit_random_cars(self):
        while True:
            if len(self.new_pop) + 1 <= pop_size:
                random_car = Car("car7")
                random_car.car_type = CarType.RANDOM
                self.new_pop.append(random_car)
            else:
                break

    def mutate_fastest(self, amount):
        print("checkpoints: " + str(self.fastest_car.checkpoint_times))
        print("dist to next: " + str(self.fastest_car.dist_to_next_checkpoint))
        print("fitness: " + str(self.fastest_car.fitness))
        print("velocity: " + str(self.fastest_car.vel))
        for i in range(amount):
            mutated_car = self.mutate(self.fastest_car)
            self.new_pop.append(mutated_car)

    def sort_cars(self):
        self.cars.sort(key=lambda car: car.fitness, reverse=True)

    def get_new_gen(self):
        self.set_total_fitness()
        self.sort_cars()
        self.set_cars_prob()
        self.set_cum_prob()
        self.best_to_new_pop(amount = int(pop_size * 0.02))
        self.crossover(amount = int(pop_size * 0.4))
        self.mutate_fastest(amount = int(pop_size * 0.4))
        self.fill_new_pop_wit_random_cars()

        return self.new_pop

