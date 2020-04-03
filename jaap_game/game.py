import pygame as pg
import math
import random
import numpy as np
import time

pg.init()

pg.font.init()
font = pg.font.SysFont('segoeui', 35)

screen_width = 1300
screen_height = 800
screen = pg.display.set_mode((screen_width, screen_height)) 

def sigmoid(s):
        return 1/(1+np.exp(-s))

def check_rect_collision(rect, objects):
    for i in range(len(objects)):
        if rect.colliderect(objects[i]):
            return i
    return -1

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
        self.spriteArea = (self.counterX * self.width, 
                             self.counterY * self.height,
                             self.width, self.height)
        self.counterX += 1
        if self.counterX == self.spritesPerRow:
            self.counterX = 0
            self.counterY += 1
        if self.counterY == self.spritesPerCol:
            if self.repeat:
                self.counterX = 0
                self.counterY = 0

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
        self.nr_of_inputs = 2
        self.nr_of_outputs = 2
        self.weights1 = np.random.rand(self.nr_of_inputs,5)
        self.weights2 = np.random.rand(5,self.nr_of_outputs)

    def feedforward(self, x):
        self.layer1 = sigmoid(np.dot(x, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return np.argmax((self.output[0]))

class Car():
    def __init__(self, colour, car_img):
        self.x = random.randrange(290, 310)
        self.y = random.randrange(95, 105)
        self.angle = 90
        self.vel = 15
        
        self.image = pg.image.load("images/cars/"+car_img+".png")
        self.rect = self.image.get_rect()
        self.rect.center = self.x, self.y
        self.arms = []
        self.arms_visible = False
        self.visible = True
        self.show_collision_rect = False
        self.can_drive = True
        self.is_crashed = False
        self.colour = colour
        self.explosion_sheet = Sprite(pg.image.load("images/explosion.png"), 200, 240, 0, False)
        self.fire_sheet = Sprite(pg.image.load("images/fire.png"), 20, 20, 0, True)

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
                total_fitness += (1/(self.checkpoint_times[i][1] - self.checkpoint_times[i - 1][1])) * 2

        total_fitness += 2*(1/self.dist_to_next_checkpoint)

        if self.nr_of_wrong_checkpoints > 0:
            total_fitness = 0
        self.fitness = total_fitness

    def drive(self, walls):
        self.rect.center = self.x, self.y
        if check_rect_collision(self.rect, walls) != -1 or not self.can_drive:
            self.is_crashed = True
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
                if self.arms_visible and self.can_drive:
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
        self.mutation_rate = 0.1

        for car in cars:
            if not car.can_drive:
                self.number_finished += 1
            if car.is_crashed:
                self.number_crashed += 1

    def set_fitness(self):
        total_fit = 0
        for i in range(pop_size):
            self.cars[i].set_fitness()
            total_fit += self.cars[i].fitness
        self.total_fitness = total_fit
        print("total fit: "+str(total_fit))

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

            nr_of_inputs = parent1.brain.nr_of_inputs
            nr_of_outputs = parent1.brain.nr_of_outputs

            child.brain.weights1 = np.concatenate((parent2.brain.weights1[mid_w1:], parent1.brain.weights1[:mid_w1]), axis=None).reshape((nr_of_inputs,5))
            child.brain.weights2 = np.concatenate((parent1.brain.weights2[mid_w1:], parent2.brain.weights2[:mid_w1]), axis=None).reshape((5,nr_of_outputs))
            
            if self.mutation_rate > random.random():
                self.new_pop.append(self.mutate(child))
            else:
                self.new_pop.append(child)
            
    def mutate(self, child):
        random_index = random.randrange(0, len(child.brain.weights1))
        random_index2 = random.randrange(0, len(child.brain.weights2))
        child.brain.weights1[random_index] = random.random()
        child.brain.weights2[random_index2] = random.random()
        child.colour = pg.Color(1,255,1)
        child.image = pg.image.load("images/cars/car3.png")
        return child

    def best_to_new_pop(self, amount):
        best_cars = cars[:amount]
        for car in best_cars:
            new_car = Car(pg.Color(128,128,255), "car7")
            new_car.brain.weights1 = car.brain.weights1
            new_car.brain.weights2 = car.brain.weights2
            self.new_pop.append(new_car)

    def sort_cars(self):
        self.cars.sort(key=lambda car: car.fitness, reverse=True)

    def get_new_gen(self):
        self.set_fitness()
        self.sort_cars()
        self.set_cars_prob()
        self.set_cum_prob()
        self.best_to_new_pop(amount = int(pop_size/40))
        self.matrix_to_array()
        self.crossover()    

        return self.new_pop
    
#population and map details
checkpoints = [
    Checkpoint(x = 400, y = 20, width = 25, height = 180, focus_off_x = 10, focus_off_y = 100, nr = 1),
    Checkpoint(x = 950, y = 20, width = 25, height = 180, focus_off_x = 10, focus_off_y = 170, nr = 2),
    Checkpoint(x = 975, y = 575, width = 305, height = 25, focus_off_x = 10, focus_off_y = 10, nr = 3),
    Checkpoint(x = 327, y = 600, width = 25, height = 180, focus_off_x = 10, focus_off_y = 10, nr = 4),
    Checkpoint(x = 10, y = 210, width = 305, height = 25, colour = pg.Color(120,0,120), focus_off_x = 280, focus_off_y = 10, nr = 5)]
walls = [
    Wall(screen_width/2, 0, screen_width, 90),
    Wall(screen_width/2, screen_height, screen_width, 90),
    Wall(0, screen_height/2, 90, screen_height),
    Wall(screen_width, screen_height/2, 90, screen_height),
    Wall(screen_width/2, screen_height/2, screen_width/2, screen_height/2),
    Wall(screen_width/1.5, 0, 90, screen_height/6),
    Wall(screen_width/2.2, 190, 90, screen_height/6),
    Wall(screen_width/2, 620, 90, screen_height/6),
    Wall(screen_width/2.7, 750, 90, screen_height/6),
    Wall(300, 550, 250, 60),
    Wall(50, 730, 250, 100),
    Wall(50, 400, 250, 60),
    Wall(300, 270, 250, 60)]

pop_size = 250
total_gens = 1
start_time = time.time()
new_gen_time = 20
delay = 10
cars = []
user_car = Car(pg.Color(255,100,255), "car4")
user_car.arms_visible = True

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

    handle_user_car(pg.key.get_pressed(), user_car)

    for car in cars:
        car.set_and_draw_sonar_arms(nr_arms = 2, arms_scan_range = 100, number_of_points = 5, distance = 60, size = 4)
        car.set_current_checkpoint_and_distance(checkpoints)
        car.rotate_blit()
        car.drive(walls)

    ga = GA(cars)
    if ((time.time() - start_time) / total_gens) > new_gen_time:
        total_gens += 1
        
        new_pop = ga.get_new_gen()
        cars = new_pop

        user_car.x = 300
        user_car.y = 100
        user_car.angle = 90

    text = " Gen: "+str(total_gens)+" Finished: "+str(ga.number_finished)+"/"+str(pop_size)
    screen.blit(font.render(text, 1, (0,0,0)),(55,40))

    pg.display.flip()
pg.quit()
