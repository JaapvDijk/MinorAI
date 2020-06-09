import pygame as pg
import math
import random
import numpy as np
import time
import datetime
import glob
from enum import Enum
from enums import *
from helpers import *
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

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
    def __init__(self, new_round_time, min_time_to_reach_first_checkpoint, car_spawn_x = 250, car_spawn_y = 120):
        self.checkpoints = [
        #Checkpoint(x = 490, y = 40, width = 50, height = 160, focus_off_x = 10, focus_off_y = 140, nr = 1),
        #Checkpoint(x = 610, y = 40, width = 30, height = 160, focus_off_x = 10, focus_off_y = 100, nr = 1),
        Checkpoint(x = 780, y = 45, width = 30, height = 95, focus_off_x = 10, focus_off_y = 75, nr = 1),
        Checkpoint(x = 1020, y = 40, width = 30, height = 160, focus_off_x = 10, focus_off_y = 130, nr = 2),
        Checkpoint(x = 1055, y = 575, width = 200, height = 30, focus_off_x = 15, focus_off_y = 10, nr = 3),
        Checkpoint(x = 315, y = 600, width = 30, height = 85, focus_off_x = 10, focus_off_y = 35, nr = 4),
        Checkpoint(x = 45, y = 560, width = 100, height = 30, focus_off_x = 90, focus_off_y = 10, nr = 5),
        Checkpoint(x = 145, y = 400, width = 100, height = 30, focus_off_x = 20, focus_off_y = 10, nr = 6),
        Checkpoint(x = 45, y = 210, width = 200, height = 30, focus_off_x = 105, focus_off_y = 10, nr = 7)
        ]
        self.walls = [
        Wall(650, 0, 1300, 90),
        Wall(650, 800, 1300, 90),
        Wall(0, 400, 90, 800),
        Wall(1300, 400, 90, 800),

        #Wall(600, 45, 80, 100), #first right
        Wall(820, 190, 180, 100), #first left
        Wall(650, 400, 1300/1.6, 400),
        Wall(850, 620, 410, 95),
        #Wall(1210, 150, 100, 300), #first brake
        Wall(200, 750, 600, 130),
        Wall(270, 570, 250, 60),
        Wall(20, 400, 250, 60),
        Wall(300, 270, 250, 60)]

        self.new_round_time = new_round_time
        self.min_time_to_reach_first_checkpoint = min_time_to_reach_first_checkpoint
        self.car_spawn_x = car_spawn_x
        self.car_spawn_y = car_spawn_y

class Level2(object):
    def __init__(self, new_round_time, min_time_to_reach_first_checkpoint, car_spawn_x = 100, car_spawn_y = 200):
        self.checkpoints = [
        Checkpoint(x = 150, y = 160, width = 20, height = 80, focus_off_x = 10, focus_off_y= 40, nr = 1),
        Checkpoint(x = 260, y = 140, width = 110, height = 20, focus_off_x = 85, focus_off_y= 10, nr = 2),
        Checkpoint(x = 420, y = 40, width = 20, height = 100, focus_off_x = 10, focus_off_y= 80, nr = 3),
        Checkpoint(x = 490, y = 160, width = 110, height = 20, focus_off_x = 60, focus_off_y = 10, nr = 4),
        Checkpoint(x = 490, y = 400, width = 110, height = 20, focus_off_x = 30, focus_off_y = 20, nr = 5)]

        self.walls = [
        Wall(1300/2, 0, 1300, 90),
        Wall(1300/2, 800, 1300, 90),
        Wall(0, 800/2, 90, 800),
        Wall(1300, 800/2, 90, 800),

        Wall(140, 150, 230, 20),
        Wall(180, 250, 440, 20),
        Wall(430, 300, 120, 320),
        Wall(250, 85, 20, 150),
        Wall(610, 200, 20, 550)]

        self.new_round_time = new_round_time
        self.min_time_to_reach_first_checkpoint = min_time_to_reach_first_checkpoint
        self.car_spawn_x = car_spawn_x
        self.car_spawn_y = car_spawn_y

class Car(object):
    def __init__(self, car_img, car_spawn_x, car_spawn_y):
        self.car_spawn_x = car_spawn_x
        self.car_spawn_y = car_spawn_y
        self.with_accel_and_braking = True
        self.x = self.car_spawn_x
        self.y = self.car_spawn_y

        self.angle = 90
        self.vel = 7
        self.max_speed = 7 # <= 15 is ez to train
        self.brake_speed = -1
        self.drive_speed = 1
        self.id = id(self)
        self.car_type = CarType.CROSSOVER
        
        self.image = pg.image.load("jaap_game/images/cars/"+car_img+".png")
        self.rect = self.image.get_rect()
        self.rect.center = self.x, self.y
        self.arms = []
        self.hide_car_arms = True
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
        self.drive_trail = [] #to be removed, temp

        self.current_checkpoint_nr = 0
        self.nr_of_wrong_checkpoints = 0
        self.dist_to_next_checkpoint = -1
        self.next_checkpoint = Checkpoint(0,0,0,0,0,0,0)
        self.laps_finished = 0
        self.finish_time = 0
        self.checkpoint_times = []

        self.brain = NeuralNetworkGenetic()

    def drive_left(self):
        self.angle += 5

    def drive_right(self):
        self.angle += -5

    def drive_forward(self):
        if self.with_accel_and_braking:
            if self.vel + self.drive_speed <= self.max_speed:
                self.increase_speed(self.drive_speed)

                if self.visible:
                    pg.draw.rect(screen, (0,0,200), [self.rect.x+15, self.rect.y+8, 15, 15]) #to be removed
        else:
            self.vel = self.max_speed

    def brake(self):
        if (self.vel + self.brake_speed) > abs(self.max_speed/4):
            self.increase_speed(self.brake_speed)

            if self.visible:
                pg.draw.rect(screen, (255,100,0), [self.rect.x, self.rect.y+8, 15, 15]) #to be removed

    def glide(self):
        if self.vel + self.brake_speed / 8 >= 0:
            self.increase_speed(self.brake_speed / 8)

            # if self.visible:
            #     pg.draw.rect(screen, (0,0,200), [self.rect.x, self.rect.y+8, 15, 15]) #to be removed

    def increase_speed(self, amount):
        self.vel += amount

    def set_new_position(self, vel):
        self.x += vel*math.sin(math.radians(self.angle))
        self.y += vel*math.cos(math.radians(self.angle))

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
        arms_spread = 0
        added_back_arm = False

        for i in range(nr_arms):
            arm_points = []
            point_x = self.x
            point_y = self.y

            arms_spread = arms_scan_range / (nr_arms - 1)
            angle = ((self.angle) + arms_scan_range / 2) - arms_spread * i
            if added_back_arm:
                arms_spread = arms_scan_range / (nr_arms - 2)
                angle = ((self.angle) + arms_scan_range / 2) - arms_spread * (i-1)

            if add_back_arm:
                angle = self.angle + 180
                arms_spread = arms_scan_range / nr_arms
                add_back_arm = not add_back_arm
                added_back_arm = True

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

    def set_current_checkpoint_and_distance(self, checkpoints, game_time, game_start_time):
        self.rect.center = self.x, self.y
        for checkpoint in checkpoints:
            self.next_checkpoint = checkpoint
            if check_rect_collision(self.rect, [checkpoint.rect]) != -1:
                if self.current_checkpoint_nr + 1 == checkpoint.nr or self.current_checkpoint_nr == len(checkpoints) and not self.is_finished:
                    self.current_checkpoint_nr = checkpoint.nr
                    self.checkpoint_times.append([checkpoint.nr, game_time - game_start_time])
                    if self.current_checkpoint_nr == len(checkpoints):
                        self.finish_time = game_time - game_start_time
                        self.next_checkpoint = checkpoints[0]
                        self.laps_finished += 1
                        self.can_drive = False
                        self.is_finished = True
                elif not self.current_checkpoint_nr == checkpoint.nr:
                    self.nr_of_wrong_checkpoints += 1

        if self.current_checkpoint_nr + 1 < len(checkpoints):
             self.next_checkpoint = checkpoints[self.current_checkpoint_nr]
        self.dist_to_next_checkpoint = math.hypot(self.x -  self.next_checkpoint.focus_point[0], self.y -  self.next_checkpoint.focus_point[1])

    def handle_user_input(self, keys, walls):
        if check_rect_collision(self.rect, walls) == -1:
            if keys[pg.K_LEFT]:
                self.drive_left()
            if keys[pg.K_RIGHT]:
                self.drive_right()
            if keys[pg.K_UP]:
                self.drive_forward()
            if keys[pg.K_DOWN]:
                self.brake()
            self.set_new_position()
        else:
            self.explosion_sheet.update()
            screen.blit(self.explosion_sheet.sheet, (self.x - self.explosion_sheet.width/2, self.y - self.explosion_sheet.height), self.explosion_sheet.spriteArea)
        
    def draw_drive_trail(self):
        for trail_point in self.drive_trail:
            pg.draw.rect(screen, (255,255,255), [trail_point[0], trail_point[1], 2, 2])

class Game(object):
    def __init__(self, level, delay, hide_car_arms):
        pg.init()
        pg.font.init()
        self.level = level
        self.delay = delay
        self.ticks = 1
        self.game_time = 0
        self.game_start_time = 0
        self.hide_car_arms = True
        self.user_car = Car("car4", self.level.car_spawn_x, self.level.car_spawn_y)
        self.user_car.car_type = CarType.USER
        self.stats_font = pg.font.SysFont('segoeui', 15)
        self.checkpoint_font = pg.font.SysFont('segoeui', 20)
        self.play_slomo = False

    def draw_walls(self):
        for wall in self.level.walls:
            pg.draw.rect(screen, wall.colour, [wall.rect.x, wall.rect.y, wall.rect.width, wall.rect.height])

    def draw_checkpoints(self):
        for checkpoint in self.level.checkpoints:
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

    def handle_user_car(self, keys, user_car, walls):
        self.user_car.handle_user_input(keys, walls)
        self.user_car.set_current_checkpoint_and_distance(self.level.checkpoints, self.game_time, self.game_start_time)
        self.user_car.set_and_draw_sonar_arms(nr_arms = 10, arms_scan_range = 180, number_of_points = 20, distance = 20, size = 2, add_back_arm = True, walls = self.level.walls)
        self.user_car.rotate_blit()

        # user_car.drive_trail.append([user_car.x, user_car.y]) #to be removed
        # user_car.draw_drive_trail() #to be removed

class Genectic_game(Game):
    def __init__(self, level, delay, hide_car_arms = True, print_fastest_car_info = False, draw_crashed_and_idling_cars = False):
        Game.__init__(self, level, delay, hide_car_arms)
        self.cars = []
        self.weights_directory = 'jaap_game/weights/'+str(time.time())+'/'
        self.fastest_finish_times = []
        self.best_fitness = []
        self.ga = GA(self.cars, self.level.car_spawn_x, self.level.car_spawn_y)
        self.draw_crashed_and_idling_cars = draw_crashed_and_idling_cars
        self.print_fastest_car_info = print_fastest_car_info
        self.draw_car_fitness = False
        self.clicked_car = GenecticCar("car6", 0, 0)
        self.drive_trail_hist = [[[0,0]]]
        self.best_direction_hist = []
        self.current_best_car = self.ga.fastest_car #todo

        for i in range(self.ga.pop_size):
            self.cars.append(GenecticCar("car7", self.level.car_spawn_x, self.level.car_spawn_y))

    def handle_cars(self, cars, walls, checkpoints):
        for car in self.cars:
            if not car.is_crashed and not car.is_finished or car.is_best or self.draw_crashed_and_idling_cars:

                should_have_reached_checkpoint_one = self.game_time - self.game_start_time > self.level.min_time_to_reach_first_checkpoint
                car_has_reached_checkpoint_one = len(car.checkpoint_times) < 1
                car_has_bad_fitness = car.fitness == 0

                if not (should_have_reached_checkpoint_one and (car_has_reached_checkpoint_one or car_has_bad_fitness)):
                    car.set_and_draw_sonar_arms(nr_arms = 10, arms_scan_range = 180, number_of_points = 30, distance = 15, size = 2, add_back_arm = True, walls = self.level.walls)
                    car.set_current_checkpoint_and_distance(self.level.checkpoints, self.game_time, self.game_start_time)
                    car.rotate_blit()
                    car.drive(self.level.walls)
                    car.set_fitness()
                else:
                    car.is_idle = True

            if self.draw_car_fitness and car.visible or car.is_best or car.is_finished:
                car_text = str(round(car.fitness,2))
                screen.blit(pg.font.SysFont('segoeui', 15).render(car_text, 1, pg.color.THECOLORS["white"]),(car.x - car.rect.width /2,car.y+10))
            
            if car.car_type == CarType.BEST:
                self.current_best_car = car
                car.drive_trail.append([car.x, car.y])
                car.draw_drive_trail()
    
    def run(self):
        run = True
        while run:
            if self.play_slomo:
                pg.time.delay(1000)
            pg.time.delay(self.delay)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
 
            screen.fill((85,96,91))

            #"Best" driving line
            trail_img = pg.image.load("jaap_game/images/best_trail.png") #todo
            screen.blit(trail_img, trail_img.get_rect())

            self.draw_walls()
            self.draw_checkpoints()
            self.draw_drive_trail_hist()
            self.draw_finish(tile_rows = 2)

            self.ticks += 1
            self.game_time = self.ticks * (self.delay / 1000)
            
            self.handle_cars(self.cars, self.level.walls, self.level.checkpoints)
            #self.handle_cars([saved_ai_car]) #todo

            #self.handle_user_car(pg.key.get_pressed(), self.user_car, self.level.walls)
            
            self.handle_commands(pg.key.get_pressed())

            self.ga.update()

            should_start_new_round = (self.game_time - self.game_start_time) > self.level.new_round_time or self.ga.all_cars_done_with_training or self.ga.number_finished > self.ga.pop_size/20 #or ga.fastest_car.is_finished or ga.fastest_car.is_crashed:
            if should_start_new_round:

                self.ga.total_gens += 1

                self.game_start_time = self.game_time

                try:
                    if self.best_fitness[-1] > self.best_fitness[-2] and self.ga.fastest_car.current_checkpoint_nr > 0: #self.fastest_finish_times[-1] < self.fastest_finish_times[-2] or 
                        self.drive_trail_hist.append(self.current_best_car.drive_trail)
                except IndexError:
                    pass

                self.cars = self.ga.get_new_gen()

                if self.ga.fastest_car.finish_time > 0:
                    self.fastest_finish_times.append(round(self.ga.fastest_car.finish_time,5))
                else:
                    self.fastest_finish_times.append(np.nan)

                self.best_fitness.append(round(self.ga.fastest_car.fitness,8))
                self.best_direction_hist.append(self.ga.fastest_car.direction_hist)

                save_fastest_car_to_file(self.ga, self.weights_directory)

                self.user_car = Car("car4", self.level.car_spawn_x, self.level.car_spawn_y)
                # self.saved_ai_car = GenecticCar("car8", self.car_spawn_x, self.car_spawn_y) #todo
                # self.saved_ai_car = get_saved_car_brain(self.saved_ai_car) #todo

                self.ga.new_pop = []
                self.ga.cars = self.cars
                
                if self.print_fastest_car_info:
                    print("checkpoints: " + str(self.ga.fastest_car.checkpoint_times))
                    print("dist to next: " + str(self.ga.fastest_car.dist_to_next_checkpoint))
                    print("next checkpoint: " + str(self.ga.fastest_car.next_checkpoint.nr))
                    print("curr checkpoint: " + str(self.ga.fastest_car.current_checkpoint_nr))
                    print("velocity on crash/finish: " + str(self.ga.fastest_car.vel))
                    print("nr of wrong checkpoint: " + str(self.ga.fastest_car.nr_of_wrong_checkpoints))
                    print("fitness: " + str(self.ga.fastest_car.fitness))
                    print("id: " + str(self.ga.fastest_car.id))
                    print("round: " + str(self.ga.total_gens))
            
            text = "Gen: "+str(self.ga.total_gens)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,45))
            text = "Finished: "+str(self.ga.number_finished)+"/"+str(self.ga.pop_size)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,58))
            text = "Crashed: "+str(self.ga.number_crashed)+"/"+str(self.ga.pop_size) 
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,71))
            text = "Idles deleted: " + str(self.ga.number_idle)+"/"+str(self.ga.pop_size)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,84))
            text = "Mutations crashed: " + str(self.ga.number_mutation_crashed)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,97))
            text = "Crossovers crashed: " + str(self.ga.number_crossover_crashed)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,110))
            text = "Fastest car id: " + str(self.ga.fastest_car.id)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,123))
            text = "New round in: " + str(round(self.level.new_round_time - (self.game_time - self.game_start_time), 1)) + " sec"
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,136))
            fastest_time = np.nan
            if len(self.fastest_finish_times) > 0:
                fastest_time = self.fastest_finish_times[-1]
            text = "last round best finish time: " + str(fastest_time) + " sec"
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,149))
            text = "Best trail length: " + str(len(self.drive_trail_hist[-1]))
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,162))

            screen.blit(pg.image.load("jaap_game/images/cars/car2.png"), (60,700))
            screen.blit(self.stats_font.render("Crossover", 1, pg.color.THECOLORS["black"]),(83,705))
            screen.blit(pg.image.load("jaap_game/images/cars/car3.png"), (150,700))
            screen.blit(self.stats_font.render("Mutation", 1, pg.color.THECOLORS["black"]),(173,705))
            screen.blit(pg.image.load("jaap_game/images/cars/car4.png"), (240,700))
            screen.blit(self.stats_font.render("Player", 1, pg.color.THECOLORS["black"]),(263,705))
            screen.blit(pg.image.load("jaap_game/images/cars/car6.png"), (330,700))
            screen.blit(self.stats_font.render("Best", 1, pg.color.THECOLORS["black"]),(353,705))
            screen.blit(pg.image.load("jaap_game/images/cars/car7.png"), (420,700))
            screen.blit(self.stats_font.render("Random", 1, pg.color.THECOLORS["black"]),(443,705))

            pg.draw.rect(screen, (0,0,0), [60, 750, 2, 2])
            pg.draw.rect(screen, (0,0,0), [65, 750, 2, 2])
            pg.draw.rect(screen, (0,0,0), [70, 750, 2, 2])
            pg.draw.rect(screen, (0,0,0), [75, 750, 2, 2])
            pg.draw.rect(screen, (0,0,0), [80, 750, 2, 2])
            screen.blit(self.stats_font.render("Best path", 1, pg.color.THECOLORS["black"]),(90,740))

            pg.draw.rect(screen, (50,50,50), [60, 775, 2, 2])
            pg.draw.rect(screen, (100,100,100), [65, 775, 2, 2])
            pg.draw.rect(screen, (150,150,150), [70, 775, 2, 2])
            pg.draw.rect(screen, (200,200,200), [75, 775, 2, 2])
            pg.draw.rect(screen, (255,255,255), [80, 775, 2, 2])
            screen.blit(self.stats_font.render("Path history (Darker = older)", 1, pg.color.THECOLORS["black"]),(90,765))

            #text = "Hotkeys: Hide (R)andoms, (C)rossovers, (B)est, (M)utations - (S)tats - (F)itness S(l)omo (D)riving_trails" #todo
            #screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["black"]),(510,755))
            text = "Clicked car " +  " vel " + str(self.clicked_car.vel) + " fit " + str(self.clicked_car.fitness)
            screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["black"]),(510,770))      

            pg.display.flip()
        pg.quit()

    def handle_commands(self, keys):
        if keys[pg.K_s]:
            show_stats_plot(self.best_fitness, self.fastest_finish_times, self.best_direction_hist)
        if keys[pg.K_b]:
            self.ga.hide_best_cars = not self.ga.hide_best_cars
        if keys[pg.K_m]:
            self.ga.hide_mutation_cars = not self.ga.hide_mutation_cars
        if keys[pg.K_c]:
            self.ga.hide_crossover_cars = not self.ga.hide_crossover_cars
        if keys[pg.K_r]:
            self.ga.hide_random_cars = not self.ga.hide_random_cars   
        if keys[pg.K_f]:
            self.draw_car_fitness = not self.draw_car_fitness
        if keys[pg.K_l]:
            self.play_slomo = not self.play_slomo
        if keys[pg.K_d]:
            self.ga.hide_driving_trail_history = not self.ga.hide_driving_trail_history
        if pg.mouse.get_pressed()[0]:
            mouse_pos = pg.mouse.get_pos()
            clicked_car =  check_car_click_collision(mouse_pos, self.ga.cars)
            self.clicked_car = self.ga.cars[clicked_car]

    def draw_drive_trail_hist(self):
        if not self.ga.hide_driving_trail_history:
            nr_of_checkpoints = len(self.drive_trail_hist)

            for i, trail in enumerate(self.drive_trail_hist):
                colour_strengt = (i+1) * (255 / nr_of_checkpoints)
                if (i+1) < nr_of_checkpoints:
                    for trail_point in trail:        
                        pg.draw.rect(screen, (colour_strengt,colour_strengt,colour_strengt), [trail_point[0], trail_point[1], 1, 1])
                else:
                    for trail_point in trail:              
                        pg.draw.rect(screen, (colour_strengt,colour_strengt,colour_strengt), [trail_point[0], trail_point[1], 2, 2])


class ReinforcementGame(Game):
    def __init__(self, level, delay, hide_car_arms):
        Game.__init__(self, level, delay, hide_car_arms)
        self.car = ReinforcementCar("car8", self.level.car_spawn_x, self.level.car_spawn_y)
    
    def step(self,action):
        pg.time.delay(self.delay)

        self.ticks += 1
        self.game_time = self.ticks * (self.delay / 1000)

        screen.fill((85,96,91))

        self.draw_walls()
        self.draw_checkpoints()
        self.draw_finish(tile_rows = 2)

        if self.car.is_crashed or (self.game_time - self.game_start_time) >= self.level.new_round_time:
            self.game_start_time = self.game_time
            self.car = ReinforcementCar("car8", self.level.car_spawn_x, self.level.car_spawn_y)

        #self.handle_car(action, self.car)
        self.car.handle(action, self.level.walls, self.level.checkpoints, self.game_time, self.game_start_time)

        text = "Gametime: " + str(round(self.game_time, 1))
        screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,70))
        text = "Next round in: " + str(round(self.level.new_round_time - (self.game_time - self.game_start_time), 1))
        screen.blit(self.stats_font.render(text, 1, pg.color.THECOLORS["white"]),(55,83))

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
                
class NeuralNetworkGenetic(object):
    def __init__(self):
        self.nr_of_inputs = 10
        self.nr_neurons_w1 = 12
        self.nr_neurons_w2 =  8
        self.nr_of_outputs = 3

        self.weights1 = np.random.rand(self.nr_of_inputs,self.nr_neurons_w1)
        self.weights2 = np.random.rand(self.nr_neurons_w1,self.nr_neurons_w2)
        self.weights3 = np.random.rand(self.nr_neurons_w2,self.nr_of_outputs)

    def feedforward(self, x):
        x = np.array(x)
        x = (x - x.min()) / (x.max() - x.min())

        self.layer1 = relu(np.dot(x, self.weights1))
        self.layer2 = relu(np.dot(self.layer1, self.weights2))
        self.output = softmax(np.dot(self.layer2, self.weights3))
        return np.argmax((self.output[0]))

class NeuralNetwork(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.b1 = np.random.rand(1,10)
        self.b2 = np.random.rand(1,5)
        self.w1 = np.random.rand(6,10)
        self.w2 = np.random.rand(10,5)

    def calculate_sigmoid(self, result):
        sig_result = 1/(1+np.exp(-result))
        return sig_result
    
    def test_sample(self,x):
        self.x = np.array([x])
        #print(self.x)
        self.feed_forward()
        return self.outcome
        
    def feed_forward(self):
        self.layer1 = self.calculate_sigmoid(np.dot(self.x, self.w1) + self.b1)
        self.outcome = self.calculate_sigmoid(np.dot(self.layer1, self.w2) + self.b2)

    def update_w2(self):
        #print(self.y)
        #print(self.outcome)
        dloss_doutcomesig = 2 *(self.y-self.outcome)
        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_dw2 = self.layer1
        dloss_dw2 = np.dot(doutcome_dw2.T, (dloss_doutcomesig * doutcomesig_doutcome))

        self.w2 += dloss_dw2
    
    def update_b2(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)
        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_db2 = 1
        dloss_db2 = dloss_doutcomesig * doutcomesig_doutcome

        self.b2 += 0.1 * dloss_db2

    def update_w1(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)

        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_dlayer1sig = self.w2
        dlayer1sig_dlayer1 = self.layer1 *(1-self.layer1)
        dlayer1_dw1 = self.x
        dloss_dw1 = np.dot(dlayer1_dw1.T, (np.dot(dloss_doutcomesig * doutcomesig_doutcome, doutcome_dlayer1sig.T) * dlayer1sig_dlayer1)) 
        
        self.w1 += dloss_dw1
    
    def update_b1(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)
        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_dlayer1sig = self.w2
        dlayer1sig_dlayer1 = self.layer1 *(1-self.layer1)
        dlayer1_db1 = 1
        dloss_db1 = np.dot(dloss_doutcomesig * doutcomesig_doutcome, doutcome_dlayer1sig.T) * dlayer1sig_dlayer1

        self.b1 += 0.1 * dloss_db1

    def feed_backward(self):
        self.update_w2()
        self.update_b2()
        self.update_w1()
        self.update_b1()
    
    def train(self, samples, outputs):
        for episodes in range(100):
            for i in range(len(samples)):
                self.x = np.array([samples[i]])
                self.y = np.array([outputs[i]])
                self.feed_forward()
                self.feed_backward()

class SupervisedGame(Game):
    def __init__(self, level, delay, hide_car_arms):
        Game.__init__(self,level, delay, hide_car_arms)
        self.df = pd.DataFrame(columns=['sensors','action'])
        self.nn = NeuralNetwork()
        self.car = SupervisedCar("car8", self.level.car_spawn_x, self.level.car_spawn_y)
    
    def check_action(self,action
    ):
        if action[pg.K_RIGHT]:
            return np.array([1,0,0,0,0])
        elif action[pg.K_LEFT]:
            return np.array([0,1,0,0,0])
        elif action[pg.K_UP]:
            return np.array([0,0,1,0,0])
        elif action[pg.K_DOWN]:
            return np.array([0,0,0,1,0])
        else:
            return np.array([0,0,0,0,1])
    
    def get_sensors(self,arms):
        sensors = []
        for arm in arms:
            sensors.append(arm.arm_length)
        sensors = np.asarray(sensors)
        return sensors
        
    def record_train_data(self,arms,action):
        sensors = self.get_sensors(arms)
        action = self.check_action(action)
        data = {'sensors':sensors,'action':action}
        self.df = self.df.append(data,ignore_index=True)
    
    def from_np_array(self,array_string):
        array_string = ','.join(array_string.replace('[ ', '[').split())
        return np.array(ast.literal_eval(array_string))
    
    def train(self):

        data = pd.read_csv("train_data.csv", converters={'sensors': self.from_np_array, 'action': self.from_np_array})
        self.nn.train(data['sensors'], data['action'])
            
    def train_run(self):
        run = True
        while run:
            pg.time.delay(self.delay)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.df.to_csv("train_data.csv", encoding='utf-8', index=False)
                    run = False
                    
            screen.fill((85,96,91))

            self.draw_walls()
            self.draw_checkpoints()
            self.draw_finish(tile_rows = 2)
            #printprint(pg.key.get_pressed())
            
            #self.handle_user_car(pg.key.get_pressed(), self.user_car, self.level.walls)
            self.record_train_data(self.user_car.arms, pg.key.get_pressed())

            pg.display.flip()
        pg.quit()
    
    def run(self):
        run = True
        while run:
            pg.time.delay(self.delay)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                    
            screen.fill((85,96,91))

            self.draw_walls()
            self.draw_checkpoints()
            self.draw_finish(tile_rows = 2)
            #printprint(pg.key.get_pressed())
            self.car.set_and_draw_sonar_arms(nr_arms = 6, arms_scan_range = 180, number_of_points = 9, distance = 50, size = 4, add_back_arm = True, walls = self.level.walls)
            action = self.nn.test_sample(self.get_sensors(self.car.arms))
            self.car.handle_user_input(action, self.level.walls)

            pg.display.flip()
        pg.quit()

class GenecticCar(Car):
    def __init__(self, car_img, car_spawn_x, car_spawn_y):
        Car.__init__(self, car_img, car_spawn_x, car_spawn_y)
        self.fitness = 0
        self.prob = 0
        self.cum_prob = 0
        self.direction_hist = [0,0,0]

    def set_fitness(self):
        total_fitness = len(self.checkpoint_times)

        #checkpoint times
        # for i in range(len(self.checkpoint_times)):
        #     if i == 0:
        #         total_fitness += 1/self.checkpoint_times[i][1]
        #     else:
        #         total_fitness += (1/(self.checkpoint_times[i][1] - self.checkpoint_times[i - 1][1]))

        if self.is_finished:
            total_fitness += 100/self.finish_time

        #distance to next
        if not self.current_checkpoint_nr == len(game.level.checkpoints):
            total_fitness += -self.dist_to_next_checkpoint * 0.001
        else:
            self.dist_to_next_checkpoint = 0

        #crash velocity
        # if self.is_crashed and not self.is_finished and not self.current_checkpoint_nr == len(game.level.checkpoints) and self.current_checkpoint_nr != 0:
        #     if self.vel < 0:
        #         total_fitness += (self.vel * 0.05)
        #     else:
        #         total_fitness += (-self.vel * 0.02)

        #Angle to next checkpoint
        # dist_front_car = 30
        # front_x = self.x + int(dist_front_car*math.sin(math.radians(self.angle)))
        # front_y = self.y + int(dist_front_car*math.cos(math.radians(self.angle)))

        # dist_front_checkpoint= two_point_dist(front_x, front_y, self.next_checkpoint.focus_point[0], self.next_checkpoint.focus_point[1])
        # dist_checkpoint_car = two_point_dist(self.x, self.y, self.next_checkpoint.focus_point[0], self.next_checkpoint.focus_point[1])

        # angle_to_next_checkpoint = angles(dist_front_car, dist_checkpoint_car, dist_front_checkpoint)
        # total_fitness += -angle_to_next_checkpoint * 0.0015

        #Not going through checkpoints in order
        if self.nr_of_wrong_checkpoints > 0:
            total_fitness += -5

        self.fitness = total_fitness

    def drive(self, walls):
        self.rect.center = self.x, self.y
        if check_rect_collision(self.rect, walls) != -1 or not self.can_drive:
            self.is_crashed = True
            return
        brain_input = []
        for arm in self.arms:
            brain_input.append(arm.arm_length)
        #brain_input.append(self.vel)
        direction = self.brain.feedforward([brain_input])
        self.direction_hist[direction] += 1
        if direction == 0:
            self.drive_right()
            self.set_new_position(self.vel - 2)
        if direction == 1:
            self.drive_left()
            self.set_new_position(self.vel - 2)
        if direction == 2: #Do nothing
            self.set_new_position(self.vel)
        # if direction == 3:
        #     self.drive_forward()
        # if direction == 4 and self.with_accel_and_braking:
        #     self.brake()
        #self.set_new_position()

    def reset_car(self):
        self.x = self.car_spawn_x
        self.y = self.car_spawn_y
        self.angle = 90

class SupervisedCar(Car):
    def __init__(self, car_img, car_spawn_x, car_spawn_y):
        Car.__init__(self, car_img, car_spawn_x, car_spawn_y)
    def handle_user_input(self, action, walls):
        if check_rect_collision(self.rect, walls) == -1:
            print(action)

            if np.array_equal(action, np.array([1,0,0,0,0])):
                self.drive_right()
            elif np.array_equal(action, np.array([0,1,0,0,0])):
                self.drive_left()
            elif np.array_equal(action, np.array([0,0,1,0,0])):
                self.drive_forward()
            elif np.array_equal(action, np.array([0,0,0,1,0])):
                self.brake()       
            else:
                self.glide()
                print(1)
            self.set_new_position()

        else:
            self.explosion_sheet.update()
            screen.blit(self.explosion_sheet.sheet, (self.x - self.explosion_sheet.width/2, self.y - self.explosion_sheet.height), self.explosion_sheet.spriteArea)

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
    
    def handle(self, action, walls, checkpoints, game_time, game_start_time):
        self.rl_drive(action,walls)
        self.set_current_checkpoint_and_distance(checkpoints, game_time, game_start_time)
        self.set_and_draw_sonar_arms(nr_arms = 5, arms_scan_range = 180, number_of_points = 20, distance = 20, size = 4, add_back_arm = True, walls = walls)
        self.rotate_blit()  

class GA(object):
    def __init__(self, cars, car_spawn_x, car_spawn_y):
        self.car_spawn_x = car_spawn_x
        self.car_spawn_y = car_spawn_y
        self.pop_size = 150
        self.total_gens = 1
        self.cars = cars
        self.new_pop = []
        self.total_fitness = 0
        self.number_finished = 0
        self.number_crashed = 0
        self.number_mutation_crashed = 0
        self.number_crossover_crashed = 0
        self.number_idle = 0
        self.all_cars_done_with_training = False        
        self.mutation_rate = 0.075
        self.fastest_car = GenecticCar("car8", self.car_spawn_x, self.car_spawn_y)
        self.prev_fastest_car = GenecticCar("car8", self.car_spawn_x, self.car_spawn_y)
        self.times_same_fastest_car = 0
        self.hide_best_cars = False
        self.hide_mutation_cars = False
        self.hide_crossover_cars = False
        self.hide_random_cars = False
        self.hide_driving_trail_history = False

    def set_fastest_car(self):
        self.prev_fastest_car = self.fastest_car
        self.fastest_car = self.cars[0]
        self.fastest_car.image = pg.image.load("jaap_game/images/cars/car8.png")

    def update(self):
        self.number_crashed, self.number_crossover_crashed, self.number_mutation_crashed, self.number_idle, self.number_finished = 0,0,0,0,0
        self.set_total_fitness
        for car in self.cars:
            if not car.can_drive:
                self.number_finished += 1
            if car.is_crashed:
                self.number_crashed += 1
            if car.is_crashed and car.car_type == CarType.MUTATION:
                self.number_mutation_crashed += 1
            if car.is_crashed and car.car_type == CarType.CROSSOVER:
                self.number_crossover_crashed += 1
            if car.is_idle and not car.is_crashed:
                self.number_idle += 1

            car.visible = True
            if self.hide_best_cars and car.car_type == CarType.BEST:
                car.visible = False
            if self.hide_random_cars and car.car_type == CarType.RANDOM:
                car.visible = False
            if self.hide_mutation_cars and car.car_type == CarType.MUTATION:
                car.visible = False
            if self.hide_crossover_cars and car.car_type == CarType.CROSSOVER:
                car.visible = False

        self.all_cars_done_with_training = self.number_crashed + self.number_idle >= self.pop_size
          
    def set_total_fitness(self):
        total_fit = 0
        for i in range(self.pop_size):
            self.cars[i].set_fitness()
            total_fit += self.cars[i].fitness
        self.total_fitness = total_fit

    def set_cars_prob_by_roulette_wheel(self):
        for i in range(self.pop_size):
            if not self.total_fitness == 0:
                self.cars[i].prob = (1 / self.total_fitness) * self.cars[i].fitness

    def set_cars_prob_by_ranked(self):
        total_prob = (len(self.cars) / 2) * (len(self.cars) + 1)
        for i in range(self.pop_size):
            if not self.total_fitness == 0:
                self.cars[i].prob = (1 / total_prob) * (len(self.cars) - i)

    def set_cum_prob(self):
        cum_prob = 0
        for i in range(self.pop_size):
            cum_prob += self.cars[i].prob
            self.cars[i].cum_prob = cum_prob

    def select_parent(self): 
        rand = random.random()
        for i in range(self.pop_size):
            if rand < self.cars[i].cum_prob:
                return self.cars[i]

    def matrix_to_array(self):
        for car in self.cars:
            car.brain.weights1 = car.brain.weights1.flatten()
            car.brain.weights2 = car.brain.weights2.flatten()
            car.brain.weights3 = car.brain.weights3.flatten()

    def cut_w1(self,parent1, parent2):
        indicator = 0
        nr_weights_in = parent1.brain.nr_of_inputs
        nr_weights_out = parent1.brain.nr_neurons_w1
        cutted_array = []
        while True:
            size_cut = random.randrange(0,nr_weights_out)
            if indicator + size_cut > (nr_weights_in * nr_weights_out):
                size_cut = (nr_weights_in * nr_weights_out) - indicator
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
        nr_weights_in = parent1.brain.nr_neurons_w1
        nr_weights_out = parent1.brain.nr_neurons_w2
        cutted_array = []
        while True:
            size_cut = random.randrange(0,nr_weights_out)
            if indicator + size_cut > (nr_weights_in * nr_weights_out):
                size_cut = (nr_weights_in * nr_weights_out) - indicator
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
    
    def cut_w3(self,parent1,parent2):
        indicator = 0
        nr_weights_in = parent1.brain.nr_neurons_w2
        nr_weights_out = parent1.brain.nr_of_outputs
        cutted_array = []
        while True:
            size_cut = random.randrange(0,nr_weights_out)
            if indicator + size_cut > (nr_weights_in * nr_weights_out):
                size_cut = (nr_weights_in * nr_weights_out) - indicator
                if indicator % 2 == 0:
                    cutted_array.append(parent1.brain.weights3[indicator:indicator + size_cut])
                else:
                    cutted_array.append(parent2.brain.weights3[indicator:indicator + size_cut])
                return cutted_array
            if indicator % 2 == 0:
                cutted_array.append(parent1.brain.weights3[indicator:indicator + size_cut])
            else:
                cutted_array.append(parent2.brain.weights3[indicator:indicator + size_cut])
            indicator += size_cut
        
    def crossover(self, amount):
        self.matrix_to_array()

        for i in range(amount):
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            child = GenecticCar("car2", self.car_spawn_x, self.car_spawn_y)

            cutted_array_w1 = self.cut_w1(parent1,parent2)
            cutted_array_w2 = self.cut_w2(parent1,parent2)
            cutted_array_w3 = self.cut_w3(parent1,parent2)

            w1 = np.array([])
            for i in cutted_array_w1:
                w1 = np.concatenate((w1,i),axis=None)
            
            w2 = np.array([])
            for i in cutted_array_w2:
                w2 = np.concatenate((w2,i),axis=None)

            w3 = np.array([])
            for i in cutted_array_w3:
                w3 = np.concatenate((w3,i),axis=None)

            inputs = child.brain.nr_of_inputs
            nr_weights1 = child.brain.nr_neurons_w1
            nr_weights2 = child.brain.nr_neurons_w2
            outputs = child.brain.nr_of_outputs

            child.brain.weights1 = w1.reshape((inputs,nr_weights1))
            child.brain.weights2 = w2.reshape((nr_weights1,nr_weights2))
            child.brain.weights3 = w3.reshape((nr_weights2,outputs))

            child.color = (1,1,255)

            child.car_type = CarType.CROSSOVER

            self.new_pop.append(child)
            
    def mutate(self, car):
        new_car = GenecticCar("car3", self.car_spawn_x, self.car_spawn_y)

        new_car.brain.weights1 = new_car.brain.weights1.flatten()
        new_car.brain.weights2 = new_car.brain.weights2.flatten()
        new_car.brain.weights3 = new_car.brain.weights3.flatten()

        fast_car_brain_weights1 = car.brain.weights1.flatten()
        fast_car_brain_weights2 = car.brain.weights2.flatten()
        fast_car_brain_weights3 = car.brain.weights3.flatten()
        
        # reached_finish = self.fastest_car.is_finished
        for i in range(len(fast_car_brain_weights1)):
            weight = fast_car_brain_weights1[i]
            should_mutate = random.random() < self.mutation_rate

            if should_mutate:
                new_val = value_between_zero_and_one(random.uniform(weight-0.4,weight+0.4)) #random.uniform(weight*0.5,weight*1.5) #random.uniform(weight-0.2,weight+0.2)
                new_car.brain.weights1[i] = new_val
            else:
                new_car.brain.weights1[i] = weight

        for i in range(len(fast_car_brain_weights2)):
            weight = fast_car_brain_weights2[i]
            should_mutate = random.random() < self.mutation_rate

            if should_mutate:
                new_val = value_between_zero_and_one(random.uniform(weight-0.4,weight+0.4))
                new_car.brain.weights2[i] = new_val
            else:
                new_car.brain.weights2[i] = weight
        
        for i in range(len(fast_car_brain_weights3)):
            weight = fast_car_brain_weights3[i]
            should_mutate = random.random() < self.mutation_rate

            if should_mutate:
                new_val = value_between_zero_and_one(random.uniform(weight-0.4,weight+0.4))
                new_car.brain.weights3[i] = new_val
            else:
                new_car.brain.weights3[i] = weight

        nr_of_inputs = new_car.brain.nr_of_inputs
        nr_weight1 = new_car.brain.nr_neurons_w1
        nr_weight2 = new_car.brain.nr_neurons_w2
        nr_of_outputs = new_car.brain.nr_of_outputs

        new_car.brain.weights1 = new_car.brain.weights1.reshape((nr_of_inputs,nr_weight1))
        new_car.brain.weights2 = new_car.brain.weights2.reshape((nr_weight1,nr_weight2))
        new_car.brain.weights3 = new_car.brain.weights3.reshape((nr_weight2,nr_of_outputs))

        new_car.car_type = CarType.MUTATION

        return new_car

    def best_to_new_pop(self, amount):
        best_cars = self.cars[:amount]
        for car in best_cars:
            new_car = GenecticCar("car6", self.car_spawn_x, self.car_spawn_y)
            new_car.brain.weights1 = car.brain.weights1
            new_car.brain.weights2 = car.brain.weights2
            new_car.brain.weights3 = car.brain.weights3
            new_car.is_best = True
            new_car.car_type = CarType.BEST
            new_car.id = car.id
            self.new_pop.append(new_car)
    
    def fill_new_pop_with_random_cars(self):
        while True:
            if len(self.new_pop) + 1 <= self.pop_size:
                random_car = GenecticCar("car7", self.car_spawn_x, self.car_spawn_y)
                random_car.car_type = CarType.RANDOM
                self.new_pop.append(random_car)
            else:
                break

    def mutate_from_fastest(self, amount):
        for i in range(amount):
            mutated_car = self.mutate(self.fastest_car)
            self.new_pop.append(mutated_car)

    def sort_cars(self):
        finished_cars = []
        not_finished_cars = []
        for car in self.cars:
            if car.finish_time > 0:
                finished_cars.append(car)
            else:
                not_finished_cars.append(car)
        
        finished_cars.sort(key=lambda car: car.finish_time)
        not_finished_cars.sort(key=lambda car: car.fitness, reverse=True)
        self.cars = finished_cars + not_finished_cars
        
    def partial_reset_if_stuck(self):
        if self.prev_fastest_car.id == self.fastest_car.id:
            self.times_same_fastest_car += 1
            if self.times_same_fastest_car > 100:
                self.times_same_fastest_car = 0
                self.fill_new_pop_with_random_cars()
                print("partial reset")
                return self.new_pop
        else:
            self.times_same_fastest_car = 0  

    def get_new_gen(self):
        self.set_total_fitness()
        self.sort_cars()
        self.set_cars_prob_by_ranked()
        self.set_cum_prob()

        #self.partial_reset_if_stuck()

        self.set_fastest_car()
        self.mutate_from_fastest(amount = int(self.pop_size * 0.3))
        self.best_to_new_pop(amount = 1)
        self.crossover(amount = int(self.pop_size * 0.5))
        self.fill_new_pop_with_random_cars()

        return self.new_pop

game = Genectic_game(
    level=Level1(
        new_round_time = 14,
        min_time_to_reach_first_checkpoint=1,
        car_spawn_x = 600, #110
        car_spawn_y = 165), #200
    delay=17,
    print_fastest_car_info = True)

game.run()

#game = ReinforcementGame(level=Level1(new_round_time=8, min_time_to_reach_first_checkpoint=1, car_spawn_x=250, car_spawn_y=120), delay=30, hide_car_arms=True)
#game.step("forward")
#game.step("forward")