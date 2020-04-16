import pygame as pg
import math
import random
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os

pg.init()

pg.font.init()
font = pg.font.SysFont('segoeui', 35)

screen_width = 1300
screen_height = 800
screen = pg.display.set_mode((screen_width, screen_height)) 

def sigmoid(s):
        return 1/(1+np.exp(-s))

def relu(s):
   return np.maximum(0,s)

def softmax(s):
    expo = np.exp(s)
    expo_sum = np.sum(np.exp(s))
    return expo/expo_sum

def check_rect_collision(rect, objects):
    for i in range(len(objects)):
        if rect.colliderect(objects[i]):
            return i
    return -1

def draw_walls(walls):
    for wall in walls:
        pg.draw.rect(screen, wall.colour, [wall.rect.x, wall.rect.y, wall.rect.width, wall.rect.height])

def draw_checkpoints(checkpoints):
    for checkpoint in checkpoints:
        pg.draw.rect(screen, checkpoint.colour, [checkpoint.x, checkpoint.y, checkpoint.rect.width, checkpoint.rect.height])
        pg.draw.rect(screen, (255,255,50), [checkpoint.focus_point[0], checkpoint.focus_point[1], 5, 5])
        screen.blit(font.render(str(checkpoint.nr), 1, (0,0,0)),(checkpoint.focus_point[0],checkpoint.focus_point[1]))

def handle_user_car(keys, user_car):
    user_car.handle_user_input(keys)
    user_car.set_current_checkpoint_and_distance(checkpoints)
    user_car.set_and_draw_sonar_arms(nr_arms = 3, arms_scan_range = 80, number_of_points = 15, distance = 8, size = 4)
    user_car.rotate_blit()

def show_stats_plot():
    fig, axs = plt.subplots(3)

    axs[0].plot(range(len(fastest_finish_times)), fastest_finish_times, label='Fastest finish time')
    axs[0].xlabel = "Generation"
    axs[0].ylabel = "Fastest time"

    axs[1].plot(range(len(total_pop_fitness)), total_pop_fitness, label='Total population fitness')
    axs[1].xlabel = "Generation"
    axs[1].ylabel = "Total pop fit"

    axs[2].plot(range(len(total_pop_fitness)), total_pop_fitness, label='Total population fitness')
    axs[2].plot(range(len(fastest_finish_times)), fastest_finish_times, label='Fastest finish time')
    axs[2].xlabel = "Generation"
    axs[2].ylabel = "Tot fit vs fastest time"

    plt.legend()
    plt.show()

def save_fastest_car_to_file(ga):
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)

    file_name = 'gen=' + str(total_gens) + ' time=' + str(ga.fastest_car.finish_time) + ' id=' + str(ga.fastest_car.id)

    text_file = open(weights_directory+file_name+".txt", "w")
    text_file.write("weights1" + "self.weights1 = np.array("+ str(ga.fastest_car.brain.weights1.reshape((ga.fastest_car.brain.nr_of_inputs, 5))) +")")
    text_file.write("weights2"+ "self.weights2 = np.array("+ str(ga.fastest_car.brain.weights2.reshape((5, ga.fastest_car.brain.nr_of_outputs))) +")")

    text_file.close()

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
    def __init__(self, x, y, width, height, colour = (0,0,0)):
        self.surf = pg.Surface((int(width), int(height))) 
        self.rect = self.surf.get_rect()
        self.rect.center = x, y
        self.colour = colour

class Checkpoint():
    def __init__(self, x, y, width, height, nr, focus_off_x, focus_off_y, colour = (150,180,150)):
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
        self.nr_of_inputs = 5
        self.nr_of_outputs = 2
        self.weights1 = np.random.rand(self.nr_of_inputs,5)
        self.weights2 = np.random.rand(5,self.nr_of_outputs)

    def feedforward(self, x):
        self.layer1 = relu(np.dot(x, self.weights1))
        self.output = softmax(np.dot(self.layer1, self.weights2))
        return np.argmax((self.output[0]))

class Car():
    def __init__(self, colour, car_img):
        self.x = 300
        self.y = 120
        self.angle = 90
        self.vel = 15
        self.id = id(self)
        
        self.image = pg.image.load("jaap_game/images/cars/"+car_img+".png")
        self.rect = self.image.get_rect()
        self.rect.center = self.x, self.y
        self.arms = []
        self.arms_visible = True
        self.visible = True
        self.show_collision_rect = False
        self.can_drive = True
        self.is_crashed = False
        self.is_finished = False
        self.is_best = False
        self.colour = colour
        self.explosion_sheet = Sprite(pg.image.load("jaap_game/images/explosion.png"), 200, 240, 0, False)
        self.fire_sheet = Sprite(pg.image.load("jaap_game/images/fire.png"), 20, 20, 0, True)

        self.current_checkpoint_nr = 0
        self.nr_of_wrong_checkpoints = 0
        self.dist_to_next_checkpoint = -1
        self.laps_finished = 0
        self.start_time = time.time()
        self.finish_time = 0
        self.checkpoint_times = []

        self.fitness = 0
        self.prob = 0
        self.cum_prob = 0

        self.brain = NeuralNetwork()

    def set_fitness(self):
        total_fitness = 0

        for i in range(len(self.checkpoint_times)):
            if i == 0:
                total_fitness += 0.2
            else:
                total_fitness += 2 * (1/(self.checkpoint_times[i][1] - self.checkpoint_times[i - 1][1]))
        if self.dist_to_next_checkpoint > 2:
            total_fitness += (1/self.dist_to_next_checkpoint)

        if self.nr_of_wrong_checkpoints > 0:
            total_fitness = 0
        self.fitness = total_fitness

    def drive(self, walls):
        self.rect.center = self.x, self.y
        if check_rect_collision(self.rect, walls) != -1 or not self.can_drive:
            self.is_crashed = True
            if self.visible:
                self.fire_sheet.update()
                screen.blit(self.fire_sheet.sheet, (self.x - self.fire_sheet.width/2, self.y - 5), self.fire_sheet.spriteArea)
                self.explosion_sheet.update()
                screen.blit(self.explosion_sheet.sheet, (self.x - self.explosion_sheet.width/2, self.y - self.explosion_sheet.height), self.explosion_sheet.spriteArea)
            return
        self.n_drive()
        
    def n_drive(self):
        arm_lengths = []
        for arm in self.arms:
            arm_lengths.append(arm.arm_length)
        direction = self.brain.feedforward([arm_lengths])
        if direction == 0:
            self.angle += 15
        if direction == 1:
            self.angle += -15
        #if direction == 2:
        self.drive_forward()
        # if direction == 3:
        #     self.drive_backwards

    def handle_user_input(self, keys):
        if check_rect_collision(self.rect, walls) == -1:
            if keys[pg.K_LEFT]:
                self.angle += 15
            if keys[pg.K_RIGHT]:
                self.angle += -15
            if keys[pg.K_UP]:
                self.drive_forward()
            if keys[pg.K_DOWN]:
                self.drive_backwards()
            if keys[pg.K_c]:
                show_stats_plot()
            if keys[pg.K_b]:
                global only_show_best_car
                only_show_best_car = not only_show_best_car
        else:
            self.explosion_sheet.update()
            screen.blit(self.explosion_sheet.sheet, (self.x - self.explosion_sheet.width/2, self.y - self.explosion_sheet.height), self.explosion_sheet.spriteArea)

    def drive_forward(self):
        self.x += int(self.vel*math.sin(math.radians(self.angle)))
        self.y += int(self.vel*math.cos(math.radians(self.angle)))

    def drive_backwards(self):
        self.x += int(-self.vel*math.sin(math.radians(self.angle)))
        self.y += int(-self.vel*math.cos(math.radians(self.angle)))

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

    def set_and_draw_sonar_arms(self, nr_arms, arms_scan_range, number_of_points, distance, size):
        self.arms = []
        self.brain.nr_of_inputs = nr_arms
        arms_spread = 0
        if nr_arms > 1:
            arms_spread = arms_scan_range / (nr_arms - 1)
        for i in range(nr_arms):
            arm_points = []
            point_x = self.x
            point_y = self.y
            angle = (self.angle + arms_scan_range / 2) - arms_spread * i
            for j in range(1, number_of_points):
                point_x += int(distance*math.sin(math.radians(angle)))
                point_y += int(distance*math.cos(math.radians(angle)))

                point_surf = pg.Surface((size, size))
                point_rect = point_surf.get_rect()
                point_rect.center = point_x, point_y
                if self.arms_visible and self.can_drive and self.visible:
                    screen.blit(point_surf, point_rect)
                if check_rect_collision(point_rect, walls) != -1:
                    break
                
                arm_points.append((point_x, point_y))
            self.arms.append(Arm(arm_points, angle - self.angle)) 

    def set_current_checkpoint_and_distance(self, checkpoints):
        self.rect.center = self.x, self.y

        for checkpoint in checkpoints:
            next_checkpoint = checkpoint
            if check_rect_collision(self.rect, [checkpoint.rect]) != -1 and self.current_checkpoint_nr + 1 is checkpoint.nr:
                self.current_checkpoint_nr = checkpoint.nr
                self.checkpoint_times.append([checkpoint.nr, time.time() - self.start_time])
                if self.current_checkpoint_nr is len(checkpoints):
                    self.finish_time = time.time() - self.start_time
                    next_checkpoint = checkpoints[1]
                    self.laps_finished += 1
                    self.can_drive = False
                    self.is_finished = True

        if self.current_checkpoint_nr + 1 < len(checkpoints):
            next_checkpoint = checkpoints[self.current_checkpoint_nr]
        self.dist_to_next_checkpoint = math.hypot(self.x - next_checkpoint.focus_point[0], self.y - next_checkpoint.focus_point[1])

class GA():
    def __init__(self, cars):
        self.cars = cars
        self.new_pop = []
        self.total_fitness = 0
        self.number_finished = 0
        self.number_crashed = 0
        self.all_have_crashed = False
        self.mutation_rate = 0.1
        self.fastest_car = self.cars[0]
        
        for car in self.cars:
            if not car.can_drive:
                self.number_finished += 1
            if car.is_crashed:
                self.number_crashed += 1
            if car.finish_time < self.fastest_car.finish_time and car.finish_time > 1 or self.fastest_car.finish_time == 0:
                self.fastest_car = car
            if only_show_best_car and not car.is_best:
                car.visible = False
            else:
                car.visible = True

        self.all_have_crashed = self.number_crashed == pop_size
            
    def set_fitness(self):
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

    def crossover(self):
        for i in range(len(self.cars) - len(self.new_pop)):
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = Car(pg.Color(255,150,1), "car2")

            mid_w1 = random.randrange(1, len(parent1.brain.weights1))
            mid_w2 = random.randrange(1, len(parent1.brain.weights2))

            weights1 = np.concatenate((np.array(parent2.brain.weights1[mid_w1:]), np.array(parent1.brain.weights1[:mid_w1])), axis=0)
            weights2 = np.concatenate((np.array(parent2.brain.weights2[mid_w2:]), np.array(parent1.brain.weights2[:mid_w2])), axis=0)

            nr_of_inputs = parent1.brain.nr_of_inputs
            nr_of_outputs = parent1.brain.nr_of_outputs

            child.brain.weights1 = weights1.reshape((nr_of_inputs,5))
            child.brain.weights2 = weights2.reshape((5,nr_of_outputs))

            if self.mutation_rate > random.random():
                self.new_pop.append(self.mutate(child))
            else:
                self.new_pop.append(child)
            
    def mutate(self, child):
        random_index = random.randrange(0, len(child.brain.weights1))
        random_index2 = random.randrange(0, len(child.brain.weights2))
        child.brain.weights1[random_index] -= random.random()
        child.brain.weights2[random_index2] -= random.random()
        child.colour = pg.Color(1,255,1)
        child.image = pg.image.load("jaap_game/images/cars/car3.png")
        return child

    def best_to_new_pop(self, amount):
        best_cars = cars[:amount]
        for car in best_cars:
            new_car = Car(pg.Color(128,128,255), "car6")
            new_car.brain.weights1 = car.brain.weights1
            new_car.brain.weights2 = car.brain.weights2
            new_car.is_best = True
            self.new_pop.append(new_car)

    def sort_cars(self):
        self.cars.sort(key=lambda car: car.fitness, reverse=True)

    def get_new_gen(self):
        self.set_fitness()
        self.sort_cars()
        self.set_cars_prob()
        self.set_cum_prob()
        self.best_to_new_pop(amount = int(pop_size/10))
        self.matrix_to_array()
        self.crossover()    

        return self.new_pop
    
#population and map details
checkpoints = [
    Checkpoint(x = 400, y = 40, width = 25, height = 160, focus_off_x = 10, focus_off_y = 85, nr = 1),
    Checkpoint(x = 950, y = 40, width = 25, height = 160, focus_off_x = 10, focus_off_y = 140, nr = 2),
    Checkpoint(x = 1055, y = 575, width = 200, height = 25, focus_off_x = 10, focus_off_y = 10, nr = 3),
    Checkpoint(x = 327, y = 600, width = 25, height = 85, focus_off_x = 10, focus_off_y = 10, nr = 4),
    Checkpoint(x = 45, y = 210, width = 200, height = 25, colour = pg.Color(120,0,120), focus_off_x = 120, focus_off_y = 10, nr = 5)]
walls = [
    Wall(screen_width/2, 0, screen_width, 90),
    Wall(screen_width/2, screen_height, screen_width, 90),
    Wall(0, screen_height/2, 90, screen_height),
    Wall(screen_width, screen_height/2, 90, screen_height),

    Wall(screen_width/2, screen_height/2, screen_width/1.6, screen_height/2),
    Wall(850, 620, 410, 95),
    Wall(200, 750, 600, screen_height/6),
    Wall(270, 570, 250, 60),
    Wall(20, 400, 250, 60),
    Wall(300, 270, 250, 60)]

pop_size = 80
total_gens = 1
start_time = time.time()
new_gen_time = 30
delay = 50
fastest_finish_times = []
total_pop_fitness = []
weights_directory = 'jaap_game/weights/'+str(time.time())+'/'
only_show_best_car = False
user_car = Car(pg.Color(255,100,255), "car4")
cars = []

for i in range(pop_size):
    cars.append(Car(pg.Color(255,150,1), "car2"))

run = True
while run:
    pg.time.delay(delay)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

    fastest_time = 999
    screen.fill((150,150,150))

    draw_walls(walls)
    draw_checkpoints(checkpoints)

    for car in cars:
        car.set_and_draw_sonar_arms(nr_arms = 5, arms_scan_range = 180, number_of_points = 5, distance = 70, size = 4)
        car.set_current_checkpoint_and_distance(checkpoints)
        car.rotate_blit()
        car.drive(walls)

    ga = GA(cars)
    if ((time.time() - start_time) / total_gens) > new_gen_time or ga.all_have_crashed:
        total_gens += 1
        
        new_pop = ga.get_new_gen()
        cars = new_pop

        if ga.fastest_car.finish_time > 0:
            fastest_finish_times.append(ga.fastest_car.finish_time)
        else:
            fastest_finish_times.append(np.nan)

        total_pop_fitness.append(ga.total_fitness / 5)

        user_car.x = 300
        user_car.y = 100
        user_car.angle = 90
        
        save_fastest_car_to_file(ga)

    handle_user_car(pg.key.get_pressed(), user_car)
    text = " Gen: "+str(total_gens)+" Finished: "+str(ga.number_finished)+"/"+str(pop_size) + " Fastest: " + str(ga.fastest_car.finish_time)
    screen.blit(font.render(text, 1, (0,0,0)),(55,40))

    pg.display.flip()
pg.quit()