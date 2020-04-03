import pygame
import math
import numpy as np
import time
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return x * (x > 0)

class NeuralNetwork(object):
    def __init__(self):
        self.weights1 = np.random.rand(5,7)
        self.weights2 = np.random.rand(7,3)
        
    def feedforward(self, x):
        self.layer1 = relu(np.dot(x, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return np.argmax((self.output[0]))

class GA(object):
    def __init__(self, players):
        self.population = players
        self.mating_pool = []
        self.new_population = []
        self.mutation_rate = 0.5
    
    def matrix_to_array(self):
        for member in self.population:
            member.brain.weights1 = member.brain.weights1.flatten()
            member.brain.weights2 = member.brain.weights2.flatten()
    
    def create_mating_pool(self):
        total_fitness = 0 
        previous_population = self.population
        for member in previous_population: 
            total_fitness+= member.fitness

        for member in previous_population:
            member.fitness_percentage = member.fitness/total_fitness
            
        self.mating_pool = previous_population

    def pick_parent(self):
        random_percentage = random.random()
        for member in self.mating_pool:
            random_percentage -= member.fitness_percentage
            if random_percentage <= 0:
                return member
    
    def select_3_fittest(self):
        self.population.sort(key=lambda player: player.fitness, reverse=True)
        old_cars = self.population[:3]
        new_cars = []
        for i in range(len(old_cars)):
            new_cars.append(Player())
            new_cars[i].brain.weights1 = old_cars[i].brain.weights1.reshape((5,7))
            new_cars[i].brain.weights2 = old_cars[i].brain.weights2.reshape((7,3))
            new_cars[i].color = (255,1,1)
        return new_cars

    def select_20_random_new_cars(self):
        new_cars = []
        for i in range(20):
            new_cars.append(Player())
        return new_cars

    def select_60_mutated_from_fittest_cars(self):
        self.population.sort(key=lambda player: player.fitness, reverse=True)
        fittest_car = self.population[0]
        new_cars = []
        for i in range(60):
            new_car = Player()
            new_car.brain.weights1 = new_car.brain.weights1.flatten()
            new_car.brain.weights2 = new_car.brain.weights2.flatten()
            random_weigth_index = random.randrange(0,len(fittest_car.brain.weights1))
            random_weigth_index2 = random.randrange(0,len(fittest_car.brain.weights2))
            random_weigth_value = random.random()
            random_weigth_value2 = random.random()
            if random.random() < self.mutation_rate:
                new_car.brain.weights1[random_weigth_index] = random_weigth_value
            else:
                new_car.brain.weights2[random_weigth_index2] = random_weigth_value2
            new_car.brain.weights1 = new_car.brain.weights1.reshape((5,7))
            new_car.brain.weights2 = new_car.brain.weights2.reshape((7,3))
            new_car.color = (1,255,1)
            new_cars.append(new_car)
        return new_cars
    
    def mutate(self, w):
        if random.random() < self.mutation_rate: 
            amount_of_mutations = random.randint(0,10)
            for i in range(amount_of_mutations):
                random_index = random.randrange(0,len(w))
                w[random_index] = w[random_index] + random.uniform(-0.1,0.1)
        return w

    def cut_w1(self,parent1, parent2):
        indicator = 0
        cutted_array = []
        while True:
            size_cut = random.randrange(0,10)
            if indicator + size_cut > 35:
                size_cut = 35 - indicator
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
            if indicator + size_cut > 21:
                size_cut = 21 - indicator
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
        
    def select_60_crossover_cars(self):
        new_cars = []
        for i in range(60):
            parent1 = self.pick_parent()
            parent2 = self.pick_parent()

            cutted_array_w1 = self.cut_w1(parent1,parent2)
            cutted_array_w2 = self.cut_w2(parent1,parent2)

            w1 = np.array([])
            for i in cutted_array_w1:
                w1 = np.concatenate((w1,i),axis=None)
            
            w2 = np.array([])
            for i in cutted_array_w2:
                w2 = np.concatenate((w2,i),axis=None)

            w1 = self.mutate(w1)
            w2 = self.mutate(w2)
            
            child = Player()
            child.brain.weights1 = w1.reshape((5,7))
            child.brain.weights2 = w2.reshape((7,3))
            child.color = (1,1,255)
            new_cars.append(child)
        return new_cars
    
    def create_new_population(self):
        self.matrix_to_array()
        crossover_cars = self.select_60_crossover_cars()
        mutated_cars = self.select_60_mutated_from_fittest_cars()
        random_cars = self.select_20_random_new_cars()
        three_fittest_cars = self.select_3_fittest()

        new_population =  crossover_cars + three_fittest_cars + random_cars
        return new_population

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((1500, 1000))

class Wall(object):
    def __init__(self, width, height,x,y):
        self.image = pygame.Surface((width, height))
        self.orginalImage = pygame.Surface((width, height))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.rect.center = x + width/2, y + height/2

class CheckPoint(Wall):
    pass

class Sensors(object):
    def __init__(self, angle, randians, player_x, player_y):
        self.image = pygame.Surface((2, 2))
        self.orginalImage = pygame.Surface((2, 2))
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

class Player(object):
    def __init__(self):
        self.image = pygame.Surface((5, 5))
        self.orginalImage = pygame.Surface((5, 5))
        self.color = (0,0,0)
        self.rect = self.image.get_rect()
        self.rect.center = 300, 300
        self.drive = True
        self.distance_to_check_point = 0
        self.time = 0
        self.distance_travelled = 0
        self.old_x = 300
        self.old_y = 300

        self.arms_length = []

        self.check_point = 0

        self.brain = NeuralNetwork()
        self.fitness = 0

        self.arms = []

        self.left_arm = Arm()
        self.left_arm.add_sensors(10,315,5, self.rect.x, self.rect.y)
        self.arms.append(self.left_arm)

        self.left_left_arm = Arm()
        self.left_left_arm.add_sensors(10,270,5, self.rect.x, self.rect.y)
        self.arms.append(self.left_left_arm)

        self.right_arm = Arm()
        self.right_arm.add_sensors(10,45,5, self.rect.x, self.rect.y)
        self.arms.append(self.right_arm)
        
        self.right_right_arm = Arm()
        self.right_right_arm.add_sensors(10,90,5, self.rect.x, self.rect.y)
        self.arms.append(self.right_right_arm)

        self.forward_arm = Arm()
        self.forward_arm.add_sensors(10,0,5, self.rect.x, self.rect.y)
        self.arms.append(self.forward_arm)
        
        self.angle = 0
    
    def normalize_arms(self):
        for i in range(len(self.arms_length)):
            self.arms_length[i] = (self.arms_length[i]/10)

    def handle_keys(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); exit()
        self.normalize_arms()
        direction = self.brain.feedforward([self.arms_length])
        if direction == 0:
           self.rotate(10)
        if direction == 1:
           self.rotate(-10)
        if direction == 2:
           self.rect.move_ip((math.sin(math.radians(self.angle)) * 10), (math.cos(math.radians(self.angle)) * 10))
           self.distance_travelled += math.sqrt(((self.rect.center[0]-self.old_x)**2) + ((self.rect.center[1] - self.old_y)**2))
           self.old_x = self.rect.x
           self.old_y = self.rect.y
           for arm in self.arms:
            for sensor in arm.sensors:
                sensor.rect.move_ip((math.sin(math.radians(self.angle)) * 10), (math.cos(math.radians(self.angle)) * 10)) 

    def rotate(self, angle):
        self.angle += angle
        if self.angle > 360:
            self.angle = 0
        elif self.angle < 0:
            self.angle = 360
        for arm in self.arms:
            for sensor in arm.sensors:
                sensor.rect.x = self.rect.x + (math.sin(math.radians(self.angle + sensor.angle)) * sensor.randians) + 10
                sensor.rect.y = self.rect.y + (math.cos(math.radians(self.angle + sensor.angle)) * sensor.randians) + 10
        self.image = pygame.transform.rotate(self.orginalImage, self.angle)

    def calc_fitness(self, check_points, start_time):
        check_point_to_reach = check_points[self.check_point]
        distance = math.sqrt(((check_point_to_reach.rect.center[0] - self.rect.x)**2) + ((check_point_to_reach.rect.center[1] - self.rect.y)**2))
        self.time = time.time() - start_time
        self.distance_to_check_point = distance
        normalized_distance_checkpoint = (self.distance_to_check_point-0)/(10000-0)
        normalized_total_distance = (self.distance_travelled-0)/(1000-0)
        normalized_time = (self.time - 0)/(100-0)
        normalized_checkpoint = (self.check_point - 0)/(len(check_points)-0)
        self.fitness =  normalized_total_distance - 0.8 * normalized_distance_checkpoint
    
    def check_collsion_sensor_wall(self, walls, arm):
        for i_sensors in range(len(arm.sensors)):
            for wall in walls:
                if arm.sensors[i_sensors].rect.colliderect(wall.rect):
                    return i_sensors
        return False
    
    def check_collsion_car_wall(self, walls):
        if self.drive:
            for wall in walls:
                if self.rect.colliderect(wall.rect):
                    self.drive = False

    def get_length_arms(self, walls):
        length_arms = []
        for arm in self.arms:
            sensor_index = self.check_collsion_sensor_wall(walls,arm)
            if sensor_index != False:
                length_arms.append(sensor_index)
            else:
                length_arms.append(len(arm.sensors))
        return length_arms

    def check_point_collision(self,check_point):
        if self.rect.colliderect(check_point.rect):
            self.check_point += 1 
   
running = True
players = []
for i in range(10):
    players.append(Player())

check_points = [CheckPoint(100,100,250,250), CheckPoint(400,10,150,500), CheckPoint(10,100,750,500),CheckPoint(200,10,1100,800)]
walls = [Wall(1500,10, 0, 0), Wall(10,980, 1490, 10), Wall(1500,10, 0, 990), Wall(10,980, 0, 10), Wall(990,500, 500, 10), Wall(980,300, 10, 600), Wall(100,400, 1400, 500), Wall(100,600,10,10)]

start_time = time.time()
while running:
    clock.tick(30)
    screen.fill((255, 255, 255))
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); exit()
    for wall in walls:
        screen.blit(wall.image, (wall.rect.x, wall.rect.y))
    for check_point in check_points:
        screen.blit(check_point.image, (check_point.rect.x, check_point.rect.y))

    for player in players:
        player.arms_length = player.get_length_arms(walls)
        player.check_collsion_car_wall(walls)
        if player.drive:
            player.handle_keys()
        player.image.set_colorkey((255, 0, 0))
        player.orginalImage.set_colorkey((255, 0, 0))
        player.image.fill(player.color)
        player.check_point_collision(check_points[player.check_point])
        for arms in player.arms:
            for sensor in arms.sensors:
                screen.blit(sensor.image, (sensor.rect.x, sensor.rect.y))
        
        screen.blit(player.image, (player.rect.x, player.rect.y))
    
    if time.time()-start_time > 15:
        running = False
        for player in players:
            player.calc_fitness(check_points, start_time)
        genetic_algortihm = GA(players)
        genetic_algortihm.create_mating_pool()
        new_population = genetic_algortihm.create_new_population()
        players = new_population
        start_time = time.time()
        running = True
    pygame.display.flip()