import pygame
import math
import numpy as np
import time
import random


width = 1300
height = 811
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)

car_surf = pygame.Surface((15, 15))
car_surf.fill(WHITE)
car_surf.set_colorkey(BLACK)

solar_arm_surf = pygame.Surface((3, 3))
solar_arm_surf.fill(WHITE)

POPULATION_SIZE = 200

def sigmoid(s):
        # activation function
        return 1/(1+np.exp(-s))

class NeuralNetwork(object):
    def __init__(self):
        self.weights1 = np.random.rand(3,5)
        self.weights2 = np.random.rand(5,3)
        #self.output = np.zeros(y.shape) 

    def feedforward(self, x):
        self.layer1 = sigmoid(np.dot(x, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return np.argmax((self.output[0]))

class Car:
    def __init__(self, color):
        self.x = 600
        self.y = 102
        self.width = 15
        self.height = 15
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.rect.center = self.x, self.y
        self.speed = 12
        self.rotation = 90
        self.image = car_surf
        self.image.fill(color)
        self.image.set_colorkey(BLACK)
        self.rotation_speed = 7
        self.brain = NeuralNetwork()
        self.drive = True
        self.fitness = 0
        self.current_checkpoint = 0
        self.start_time = time.time()
        self.end_time = 0
        self.checkpoint_len = len(checkpoints)
        self.next_checkpoint = checkpoints[0]
        self.finished = False
        self.arms = solar_arms


    def move_left(self):
        if(self.rotation + self.rotation_speed > 360):
            self.rotation = 0
        else:
            self.rotation += self.rotation_speed
    
    def move_right(self):
        if(self.rotation - self.rotation_speed < 0):
            self.rotation = 360
        else:
            self.rotation -= self.rotation_speed
    
    def move_up(self):
        x = int(self.speed*math.sin(math.radians(self.rotation)) + self.x)
        y = int(self.speed*math.cos(math.radians(self.rotation)) + self.y)
        rect = self.image.get_rect()
        rect.center = x, y
        for obj in objects:
            if rect.colliderect(obj):
                self.drive = False
                self.end_time = time.time() - self.start_time
                return

        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.width,self.height)

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rotRect = rotated_image.get_rect()
        rotRect.center = self.x, self.y
        screen.blit(rotated_image, rotRect)
    
    def check_checkpoint_collide(self):
        for i in range(len(checkpoints)):
            if self.rect.colliderect(checkpoints[i].rect):
                if i+1 > self.current_checkpoint:
                    self.current_checkpoint = i + 1
                    if i + 1 < len(checkpoints):
                        self.next_checkpoint = checkpoints[i + 1]

                    if i == 7:
                        self.finished = True
                        self.drive = False
                    break

    def calculate_fitness(self):
        distance = math.sqrt(((self.next_checkpoint.center[0] - self.x)**2) + ((self.next_checkpoint.center[1] - self.y)**2))
        normalized_distance = (distance/(2000))
        normalized_time = (self.end_time)/(100)
        normalized_checkpoint = (self.current_checkpoint)/(self.checkpoint_len)

        self.fitness = (normalized_checkpoint * 2) - (normalized_distance / 1.2)
        if self.fitness > 0:
            return self.fitness
        else:
            self.fitness = 0
            return self.fitness

    def replace_brain(self, new_brain):
        self.brain.weights1 = new_brain.weights1
        self.brain.weights2 = new_brain.weights2

    def flatten_brain(self):
        self.brain.weights1 = self.brain.weights1.flatten()
        self.brain.weights2 = self.brain.weights2.flatten()

    def reshape_brain(self):
        self.brain.weights1 = self.brain.weights1.reshape((3,5))
        self.brain.weights2 = self.brain.weights2.reshape((5,3))
    

class SolarArm:
    def __init__(self, arms, rotation, spread):
        self.points = arms
        self.length = len(arms)
        self.rotation = rotation
        self.spread = spread

    def draw(self, x, y, rotation, objects):
        i = 0
        self.spread = 5
        breaking = False
        for point in self.points:
            self.spread += 10
            point.x = int(self.spread*math.sin(math.radians(self.rotation + rotation)) + x)
            point.y = int(self.spread*math.cos(math.radians(self.rotation + rotation)) + y)

            point.image.get_rect().center = point.x, point.y

            rotRect = point.image.get_rect()
            rotRect.center = point.x, point.y

            for obj in objects:
                if obj.colliderect(rotRect):
                    self.length = i
                    breaking = True
                    break
            
            if breaking:
                break

            i = i + 1
            screen.blit(point.image, (point.x, point.y))

class SolarArmPoint:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 3
        self.height = 3
        self.image = solar_arm_surf

class Checkpoint:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = x + (width / 2), y + (height / 2)
        self.rect = pygame.Rect(x, y, width, height)


class GA:
    def __init__(self):
        self.cars = cars
        self.total_fitness = 0
        self.last_fitness = 0
        self.crossover_rate = 0.85
        self.mutation_rate = 0.1
        self.checkpoints = checkpoints
        self.current_generation = 1
        self.last_finished = 0
        self.mutated = 0
    
    def calculate_total_fitness(self):
        for car in self.cars:
            self.total_fitness += car.calculate_fitness()
    
    def calculate_total_finished(self):
        self.last_finished = 0
        for car in self.cars:
            if car.finished:
                self.last_finished += 1


    def refactor_car_fitness(self):
        for car in self.cars:
            car.fitness = car.calculate_fitness() / self.total_fitness

    def select_parent_cars(self):
        random_percentage = random.random()
        for car in self.cars:
            random_percentage -= car.fitness

            if random_percentage <= 0:
                return car

    def select_random_car(self):
        return self.cars[random.randrange(0, len(self.cars))]


    def crossover(self):
        parent1 = self.select_parent_cars()
        parent2 = self.select_parent_cars()

        if parent1.brain.weights1.shape != (15,):
            parent1.flatten_brain()

        if parent2.brain.weights1.shape != (15,):
            parent2.flatten_brain()

        if random.random() < self.crossover_rate:
            
            rand_index = random.randint(0, len(parent1.brain.weights1) - 1)

            parent1_half_weights1 = parent1.brain.weights1[:rand_index]
            parent2_half_weights1 = parent2.brain.weights1[rand_index:]

            parent1_half_weights2 = parent1.brain.weights2[:rand_index]
            parent2_half_weights2 = parent2.brain.weights2[rand_index:]

            child1 = Car(BLUE)
            child1.brain.weights1 = np.concatenate((parent1_half_weights1, parent2_half_weights1), axis=None).reshape((3,5))
            child1.brain.weights2 = np.concatenate((parent1_half_weights2, parent2_half_weights2), axis=None).reshape((5,3))

            child2 = Car(BLUE)
            child2.brain.weights1 = np.concatenate((parent2_half_weights1, parent1_half_weights1), axis=None).reshape((3,5))
            child2.brain.weights2 = np.concatenate((parent2_half_weights2, parent1_half_weights2), axis=None).reshape((5,3))

            return child1, child2


        new_car1 = Car(WHITE)
        new_car2 = Car(WHITE)

        new_car1.replace_brain(parent1.brain)
        new_car2.replace_brain(parent2.brain)

        new_car1.reshape_brain()
        new_car2.reshape_brain()

        return new_car1, new_car2

    def mutation(self, car):
        

        if random.random() < self.mutation_rate:
            self.mutated += 1
            new_car = Car(WHITE)

            new_car.replace_brain(car.brain)

            if new_car.brain.weights1.shape != (15,): 
                new_car.flatten_brain()

            rand_index = random.randint(0, len(new_car.brain.weights1) - 1)

            new_car.brain.weights1[rand_index] = random.random()
            new_car.brain.weights2[rand_index] = random.random()

            new_car.reshape_brain()

            return new_car

        return car

    def get_best_2_cars(self, sorted_cars):
        return sorted_cars[:2]

    def create_new_generation(self):

        self.calculate_total_fitness()
        sorted_cars = sorted(self.cars, key=lambda car: car.fitness, reverse=True)

        self.refactor_car_fitness()
        self.calculate_total_finished()

        new_cars = []
        best_2_cars = self.get_best_2_cars(sorted_cars)

        car_1 = Car(RED)
        car_1.replace_brain(best_2_cars[0].brain)

        car_2 = Car(RED)
        car_2.replace_brain(best_2_cars[1].brain)


        for i in range(int((len(self.cars) - 2) / 2)):
            new_children = self.crossover()
            new_cars.append(new_children[0])
            new_cars.append(new_children[1])
        

        for car in new_cars:
            self.mutation(car)

        print(self.mutated)
        new_cars.append(car_1)
        new_cars.append(car_2)


        self.cars = new_cars
        self.last_fitness = self.total_fitness
        self.total_fitness = 0

        self.current_generation += 1


# Generate solar arms
solar_arm_angles = [315, 0, 45]
solar_arms = []
for angle in solar_arm_angles:
    solar_arm_points = []
    for i in range(25):
        solar_arm_points.append(SolarArmPoint())
    solar_arms.append(SolarArm(solar_arm_points, angle, 10))


# Generate checkpoints on map
checkpoints = []
checkpoints.append(Checkpoint(650, 5, 20, 200))
checkpoints.append(Checkpoint(1105, 197, 200, 20))
checkpoints.append(Checkpoint(1105, 197, 200, 20))
checkpoints.append(Checkpoint(650, 400, 200, 20))
checkpoints.append(Checkpoint(1095, 600, 200, 20))
checkpoints.append(Checkpoint(5, 600, 200, 20))
checkpoints.append(Checkpoint(450, 400, 200, 20))
checkpoints.append(Checkpoint(5, 197, 200, 20))


# generate map (white borders)
objects = []
objects.append(pygame.Rect(0, 10, 1300, 11))
objects.append(pygame.Rect(205, 205, 900, 11))
objects.append(pygame.Rect(1294, 10, 11, 805))
objects.append(pygame.Rect(0, 10, 11, 805))
objects.append(pygame.Rect(850, 405, 550, 11))
objects.append(pygame.Rect(644, 205, 11, 400))
objects.append(pygame.Rect(644, 605, 450, 11))
objects.append(pygame.Rect(0, 805, 1300, 11))
objects.append(pygame.Rect(205, 605, 450, 11))
objects.append(pygame.Rect(0, 405, 450, 11))
objects.append(pygame.Rect(450, 10, 10, 205))


cars = []
run_code = True

for i in range(POPULATION_SIZE):
    cars.append(Car(WHITE))

ga = GA()

start_time = time.time()
while run_code:
    screen.fill(BLACK)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            run_code = False
    
    for checkpoint in checkpoints:
        pygame.draw.rect(screen, RED, [checkpoint.x, checkpoint.y, checkpoint.width, checkpoint.height], 0)
    
    for obj in objects:
        pygame.draw.rect(screen, WHITE, obj, 0)

    textsurface = myfont.render("generation: " + str(ga.current_generation), False, WHITE)
    screen.blit(textsurface,(20,20))
    
    textsurface = myfont.render("total fitness: " + str(ga.last_fitness), False, WHITE)
    screen.blit(textsurface,(20,50))

    textsurface = myfont.render("total finished: " + str(ga.last_finished), False, WHITE)
    screen.blit(textsurface,(20,80))

    if time.time() - start_time > 15:
        for car in ga.cars: 
            car.drive = False
            car.end_time - time.time() - car.start_time
        ga.create_new_generation()
        time.sleep(1)
        start_time = time.time()

    for car in ga.cars:           
        car.draw()
        car.check_checkpoint_collide()

        arm_lenghts = []
        for arm in car.arms:
            arm.draw(car.x, car.y, car.rotation, objects)
            arm_lenghts.append(arm.length)


        if car.drive:
            highest_index = car.brain.feedforward(np.array([arm_lenghts]))
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_UP]:
            #     car.move_up()

            # if keys[pygame.K_LEFT]:
            #     car.move_left()
            
            # if keys[pygame.K_RIGHT]:
            #     car.move_right()


            if highest_index == 0:
                car.move_left()
            elif highest_index == 1:
                car.move_right()
            else:
                car.move_up()

        else:
            continue

    pygame.display.flip()
    clock.tick(30)