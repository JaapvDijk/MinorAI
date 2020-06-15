import pygame as pg
import math
import random
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os
import glob
#from keras_neural_net import NeuralNet
from test_neural import NeuralNet
from enum import Enum
import pandas as pd
import ast


pg.init()
pg.font.init()
font = pg.font.SysFont('segoeui', 15)

screen_width = 1300
screen_height = 800




class Wall(object):
    def __init__(self, x,y, width, height):
        self.image = pg.Surface((width, height))
        self.orginalImage = pg.Surface((width, height))
        self.rect = self.image.get_rect()
        self.rect.center = x, y
        self.colour = pg.color.THECOLORS["darkgray"]

class CheckPoint(Wall):
    def __init__(self, x,y, width, height):
        super().__init__(x,y, width, height)
        self.colour = pg.color.THECOLORS["yellow"]
        self.touched = False

class FocusPoint(Wall):
    def __init__(self, x,y, width, height):
        super().__init__(x,y, width, height)
        self.colour = pg.color.THECOLORS["red"]

class Sensors(object):
    def __init__(self, angle, randians, player_x, player_y):
        self.image = pg.Surface((2, 2))
        self.orginalImage = pg.Surface((2, 2))
        self.rect = self.image.get_rect()
        self.rect.center = player_x + (math.sin(math.radians(angle)) * randians) + 10, player_y + (math.cos(math.radians(angle)) * randians) + 10
        self.angle = angle
        self.randians = randians

class Arm(object):
    def __init__(self):
        self.sensors = []
    
    def add_sensors(self,amount,angle,radians, player_x, player_y):
        for i in range(1, amount):
            self.sensors.append(Sensors(angle,radians * i *5, player_x, player_y))


class Car():
    def __init__(self, car_img, x, y):
        self.angle = 90

        self.v = 0.1
        self.a = 0.05

        self.next_checkpoint = 0 

        self.training_results_arm_0 = []
        self.training_results_arm_1 = []
        self.training_results_arm_2 = []
        self.training_results_arm_3 = []
        self.training_results_arm_4 = []
        self.training_results_velocity = []

        self.drive = True

        self.training_results = []
        self.target_data = []

        self.brain = NeuralNet()

        self.colour = pg.color.THECOLORS["black"]
        
        self.image = pg.image.load("images/cars/"+car_img+".png")
        self.orginalImage = pg.image.load("images/cars/"+car_img+".png")

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        #arms
        self.arms = []

        left_arm = Arm()
        left_arm.add_sensors(20,90,3, self.rect.x, self.rect.y)
        self.arms.append(left_arm)

        left_arm_forward = Arm()
        left_arm_forward.add_sensors(20,45,3, self.rect.x, self.rect.y)
        self.arms.append(left_arm_forward)

        forward_arm = Arm()
        forward_arm.add_sensors(20,0,3, self.rect.x, self.rect.y)
        self.arms.append(forward_arm)

        right_arm_forward = Arm()
        right_arm_forward.add_sensors(20,315,3, self.rect.x, self.rect.y)
        self.arms.append(right_arm_forward)

        right_arm = Arm()
        right_arm.add_sensors(20,270,3, self.rect.x, self.rect.y)
        self.arms.append(right_arm)
    
        
    def step(self, action):
        direction = action
        
        if direction == 0:
            self.rotate(5)
            self.update_coordinate()
        if direction == 1:
            self.rotate(-5)
            self.update_coordinate()
        elif direction == 2:
            self.v += 0.25
            self.update_coordinate()
        elif direction == 3:
            self.v -= 0.1
            self.update_coordinate()

    def check_collsion_arm_wall(self,walls):
        collided_arms = []
        for arm in self.arms:
            collided_arms.append(self.check_collision_sensor_wall(arm, walls))
        return collided_arms

    def check_collision_sensor_wall(self, arm, walls):
        for sensor_index in range(len(arm.sensors)):
            for wall in walls:
                if arm.sensors[sensor_index].rect.colliderect(wall.rect):
                    return sensor_index
        return len(arm.sensors)

    
    def check_collsion_car_wall(self, walls):
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                return True
    
    def check_distance_car_checkpoint(self, checkpoints):
        #print(self.next_checkpoint)
        return math.sqrt((checkpoints[self.get_next_checkpoint(checkpoints)].rect.center[0] - self.rect.x) **2 + (checkpoints[self.get_next_checkpoint(checkpoints)].rect.center[1] - self.rect.y) **2)
    
    #recent change
    def check_collsion_car_checkpoint(self, checkpoints):
        if self.rect.colliderect(checkpoints[self.next_checkpoint]):
            #if checkpoint.touched == False:
            checkpoints[self.next_checkpoint].touched = True
            self.next_checkpoint +=1

        # for index,checkpoint in enumerate(checkpoints):
        #     if self.rect.colliderect(checkpoint.rect):
        #         if checkpoint.touched == False:
        #             checkpoint.touched = True
        #             self.next_checkpoint +=1
                    #return (index + 1)
        #return 0
    
    def get_next_checkpoint(self, checkpoints):
        if self.next_checkpoint==len(checkpoints):
            print(1)
            for checkpoint in checkpoints:
                checkpoint.touched = False
            self.next_checkpoint = 0
        return self.next_checkpoint

    def rotate(self, angle):
        self.angle += angle
        if self.angle > 360:
            self.angle = 0
        elif self.angle < 0:
            self.angle = 360
        self.image = pg.transform.rotate(self.orginalImage, self.angle)
    
    def update_sensors(self):
         for arm in self.arms:
            for sensor in arm.sensors:
                sensor.rect.x = self.rect.x + (math.sin(math.radians(self.angle + sensor.angle)) * sensor.randians) + 10
                sensor.rect.y = self.rect.y + (math.cos(math.radians(self.angle + sensor.angle)) * sensor.randians) + 10
        

    def update_coordinate(self):
        self.rect.x += self.v * math.sin(math.radians(self.angle))
        self.rect.y += self.v * math.cos(math.radians(self.angle))
    
    def handle_user_input(self):
        if pg.key.get_pressed()[pg.K_LEFT]:
            self.rotate(5)
            if pg.key.get_pressed()[pg.K_UP]:
                self.update_coordinate()
        elif pg.key.get_pressed()[pg.K_RIGHT]:
            self.rotate(-5)
            if pg.key.get_pressed()[pg.K_UP]:
                self.update_coordinate()
        elif pg.key.get_pressed()[pg.K_UP]:
            self.update_coordinate()



        

class Env(object):
    def __init__(self):
        self.walls = [
                Wall(650, 0, 1300, 90),
                Wall(650, 800, 1300, 90),
                Wall(0, 400, 90, 800),
                Wall(1300, 400, 90, 800),
                Wall(820, 190, 180, 100), 
                Wall(650, 400, 1300/1.6, 400),
                Wall(750, 620, 410, 95),
                Wall(200, 750, 400, 130),
                Wall(270, 570, 250, 60),
                Wall(20, 400, 250, 60),
                Wall(300, 270, 250, 50)
        ]

        self.checkpoints = [

            CheckPoint(x = 350, y = 120, width = 30, height = 150),
            CheckPoint(x = 450, y = 120, width = 30, height = 150),
            CheckPoint(x = 600, y = 120, width = 30, height = 150),
            CheckPoint(x = 800, y = 100, width = 30, height = 100),
            #CheckPoint(x = 800, y = 100, width = 30, height = 150),
            CheckPoint(x = 1000, y = 120, width = 30, height = 150),
            CheckPoint(x = 1150, y = 200, width = 200, height = 30),
            CheckPoint(x = 1150, y = 350, width = 200, height = 30),
            CheckPoint(x = 1150, y = 450, width = 200, height = 30),
            CheckPoint(x = 1150, y = 550, width = 200, height = 30),
            CheckPoint(x = 1150, y = 650, width = 200, height = 30),
            #CheckPoint(x = 1000, y = 700, width = 30, height = 150),
            CheckPoint(x = 800, y = 700, width = 30, height = 150),
            CheckPoint(x = 500, y = 650, width = 30, height = 150),
            CheckPoint(x = 300, y = 630, width = 30, height = 100),
            CheckPoint(x = 200, y = 630, width = 30, height = 100),
            CheckPoint(x = 100, y = 580, width = 100, height = 30),
            CheckPoint(x = 150, y = 490, width = 200, height = 30),
            CheckPoint(x = 200, y = 400, width = 100, height = 30),
            CheckPoint(x = 100, y = 260, width = 130, height = 30),
            
           
            
            # CheckPoint(x = 145, y = 400, width = 100, height = 30),
            # CheckPoint(x = 45, y = 210, width = 200, height = 30)
        ]

        self.focus_points = []

        for checkpoint in self.checkpoints:
            focuspoint = FocusPoint(checkpoint.rect.center[0], checkpoint.rect.center[1],2,2)
            self.focus_points.append(focuspoint)



        self.agent = Car("car4", 220, 50)
        self.hide_car_arms = False
        self.run = True
        self.reward = 0
    
    def step(self, action):
        distance_before_action = self.agent.check_distance_car_checkpoint(self.checkpoints)
        self.agent.step(action)
        distance_after_action = self.agent.check_distance_car_checkpoint(self.checkpoints)
        self.agent.check_collsion_car_checkpoint(self.checkpoints)
        self.reward = -(distance_after_action - distance_before_action)
        if not self.draw():
            return
        done = self.agent.check_collsion_car_wall(self.walls)
        self.agent.update_sensors()
        arm1,arm2,arm3,arm4,arm5 = self.agent.check_collsion_arm_wall(self.walls)
        next_state = [arm1,arm2,arm3,arm4,arm5,self.agent.v]

        return next_state, self.reward, done
    
    def reset(self):
        self.agent.rect.x = 220
        self.agent.rect.y = 50
        self.agent.angle = 90
        self.reward = 0
        self.agent.v = 0
        self.agent.next_checkpoint = 0
        for checkpoint in self.checkpoints:
            checkpoint.touched = False
        arm1,arm2,arm3,arm4,arm5 = self.agent.check_collsion_arm_wall(self.walls)
        return np.array([arm1,arm2,arm3,arm4,arm5,self.agent.v])
    
    def draw(self):
        # run = True
        # screen = pg.display.set_mode((screen_width, screen_height))
        # while run:
                #self.agent.handle_user_input()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        
        pg.time.delay(17)
        screen = pg.display.set_mode((screen_width, screen_height))
        screen.fill((85,96,91))
        
        
        self.agent.update_sensors()
        self.agent.check_collsion_car_wall(self.walls)
        #self.agent.check_collsion_car_checkpoint(self.checkpoints)
        #print(self.agent.check_distance_car_checkpoint(self.checkpoints))
        #self.agent.handle_user_input()
        #print(0.0001 * self.agent.check_distance_car_checkpoint(self.checkpoints))
        screen.blit(self.agent.image, (self.agent.rect.x,self.agent.rect.y))
        for arms in self.agent.arms:
            for sensor in arms.sensors:
                screen.blit(sensor.image, (sensor.rect.x, sensor.rect.y))
        for wall in self.walls:
            pg.draw.rect(screen, wall.colour, [wall.rect.x, wall.rect.y, wall.rect.width, wall.rect.height])
        
        # for checkpoint in self.checkpoints:
        #     pg.draw.rect(screen, checkpoint.colour, [checkpoint.rect.x, checkpoint.rect.y, checkpoint.rect.width, checkpoint.rect.height])
        
        for focuspoint in self.focus_points:
            pg.draw.rect(screen, focuspoint.colour, [focuspoint.rect.x, focuspoint.rect.y, focuspoint.rect.width, focuspoint.rect.height])

        pg.draw.line(screen, pg.color.THECOLORS["green"] , (self.agent.rect.center), (self.checkpoints[self.agent.get_next_checkpoint(self.checkpoints)].rect.center), 1)
        textsurface = font.render(str(self.reward), False, (0, 0, 0))
        screen.blit(textsurface,(500,500))
        pg.display.flip()
        return True
            

env = Env()
env.draw()



    
    