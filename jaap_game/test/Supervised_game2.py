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

screen_width = 1300
screen_height = 800
screen = pg.display.set_mode((screen_width, screen_height)) 


class Wall(object):
    def __init__(self, x,y, width, height):
        self.image = pg.Surface((width, height))
        self.orginalImage = pg.Surface((width, height))
        self.rect = self.image.get_rect()
        self.rect.center = x, y
        self.colour = pg.color.THECOLORS["darkgray"]

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

        self.v = 0
        self.a = 0.05

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
        
        self.image = pg.image.load("jaap_game/images/cars/"+car_img+".png")
        self.orginalImage = pg.image.load("jaap_game/images/cars/"+car_img+".png")

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        #arms
        self.arms = []

        left_arm = Arm()
        left_arm.add_sensors(10,90,3, self.rect.x, self.rect.y)
        self.arms.append(left_arm)

        left_arm_forward = Arm()
        left_arm_forward.add_sensors(10,45,3, self.rect.x, self.rect.y)
        self.arms.append(left_arm_forward)

        forward_arm = Arm()
        forward_arm.add_sensors(10,0,3, self.rect.x, self.rect.y)
        self.arms.append(forward_arm)

        right_arm_forward = Arm()
        right_arm_forward.add_sensors(10,315,3, self.rect.x, self.rect.y)
        self.arms.append(right_arm_forward)

        right_arm = Arm()
        right_arm.add_sensors(10,270,3, self.rect.x, self.rect.y)
        self.arms.append(right_arm)

    def save_df_to_csv(self):
        #print(1)
        #print(self.training_results_arm_1)
        #print(len(self.training_results_arm_0))
        #print(len(self.training_results_arm_1))
        #print(len(self.training_results_arm_2))
        #print(len(self.target_data))
        #print(self.target_data)
        training_target_data = {'arm_1':self.training_results_arm_0, 
        'arm2':self.training_results_arm_1,
        'arm3':self.training_results_arm_2,
        'arm4':self.training_results_arm_3,
        'arm5':self.training_results_arm_4,
        'target_data': self.target_data}
        #print(training_target_data)
        df = pd.DataFrame(training_target_data) 
        
        #df = df.drop(df[df.target_data == "[1, 1, 1]"].index)
        #print(df)
        df.to_csv("wanne.csv", index=False)
    
    def load_train(self): 
        df = pd.read_csv('wanne.csv')
        df = df.drop(df[df.target_data == "[1, 1, 1]"].index)
        df.to_csv("wanne.csv", index=False)

        generic = lambda x: eval(x)
        conv = { 'target_data': generic
        }

        df = pd.read_csv('wanne.csv', converters=conv)
        #df = df.drop(df[df.target_data == [[1, 1, 1]]].index)
        #print(df["target_data"])
        return df
         
    
    def train(self):
        df = self.load_train()
        self.brain.train()
    
    def record_train(self, collided_arms):
        #print(collided_arms)
        self.training_results_arm_0.append(collided_arms[0])
        #print(self.training_results_arm_0)
        self.training_results_arm_1.append(collided_arms[1])
        #print(self.training_results_arm_1)
        self.training_results_arm_2.append(collided_arms[2])
        self.training_results_arm_3.append(collided_arms[3])
        self.training_results_arm_4.append(collided_arms[4])

        #print(self.training_results_arm_2)

        if pg.key.get_pressed()[pg.K_LEFT]:
            if pg.key.get_pressed()[pg.K_UP]:
                self.target_data.append([1,1,0])
            else:
                self.target_data.append([1,0,0])
        elif pg.key.get_pressed()[pg.K_RIGHT]:
            if pg.key.get_pressed()[pg.K_UP]:
                self.target_data.append([0,1,1])
            else:
                self.target_data.append([0,0,1])
        elif pg.key.get_pressed()[pg.K_UP]:
            self.target_data.append([0,1,0])
        else:
            self.target_data.append([1,1,1])
        
    def autonomous_drive(self, collided_arms):
        #print(self.brain.predict([[19,1,1,1,1]]))
        direction = self.brain.predict(np.array([collided_arms])).tolist()
        #print(direction)
        
        if direction == [1,1,0]:
            self.rotate(5)
            self.update_coordinate()
        if direction == [1,0,0]:
            self.rotate(5)
        elif direction == [0,1,1]:
            self.rotate(-5)
            self.update_coordinate()
        elif direction == [0,0,1]:
            self.rotate(-5)
        elif direction == [0,1,0]:
            self.update_coordinate()
        else:
            self.update_coordinate()

    def check_collsion_arm_wall(self,walls):
        collided_arms = []
        for arm in self.arms:
            collided_arms.append(self.check_collision_sensor_wall(arm, walls))
        #print(collided_arms)
        return collided_arms

    def check_collision_sensor_wall(self, arm, walls):
        for sensor_index in range(len(arm.sensors)):
            for wall in walls:
                if arm.sensors[sensor_index].rect.colliderect(wall.rect):
                    return sensor_index
        return len(arm.sensors)


    
    def check_collsion_car_wall(self, walls):
        if self.drive:
            for wall in walls:
                if self.rect.colliderect(wall.rect):
                    self.drive = False
        return
    
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

    def update_coordinate(self):
        self.rect.x += int(5 * math.sin(math.radians(self.angle)))
        self.rect.y += int(5 * math.cos(math.radians(self.angle)))

        
walls = [
        Wall(650, 0, 1300, 90),
        Wall(650, 800, 1300, 90),
        Wall(0, 400, 90, 800),
        Wall(1300, 400, 90, 800),

        #Wall(600, 45, 80, 100), #first right
        Wall(820, 190, 180, 100), #first left
        Wall(650, 400, 1300/1.6, 400),
        Wall(850, 620, 410, 95),
        #Wall(1210, 150, 100, 300), #first brake
        Wall(200, 750, 400, 130),
        Wall(270, 570, 250, 60),
        Wall(20, 400, 250, 60),
        Wall(300, 270, 250, 50)]

#Car
user_car = Car("car4", 300, 90)
user_car.train()
hide_car_arms = False

run = True

while run:
    pg.time.delay(20)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            user_car.save_df_to_csv()
            run = False

    screen.fill((85,96,91))

    for wall in walls:
        pg.draw.rect(screen, wall.colour, [wall.rect.x, wall.rect.y, wall.rect.width, wall.rect.height])

    user_car.check_collsion_car_wall(walls)
    user_car.update_sensors()
    if user_car.drive:
        user_car.autonomous_drive(user_car.check_collsion_arm_wall(walls))
        #user_car.handle_user_input()

    
    user_car.record_train(user_car.check_collsion_arm_wall(walls))
    screen.blit(user_car.image, (user_car.rect.x,user_car.rect.y))
    
    for arms in user_car.arms:
        for sensor in arms.sensors:
            screen.blit(sensor.image, (sensor.rect.x, sensor.rect.y))

    pg.display.flip()
pg.quit()