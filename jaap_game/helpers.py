from __future__ import division
import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
from matplotlib import markers
import os
from math import acos, degrees, sqrt
from sklearn.preprocessing import StandardScaler
import pickle
import math
from enums import CarType

def tanh(x):
    np.asarray(x)
    return np.tanh(x)
    
def sigmoid(x):
        return 1/(1+np.exp(-x))

def relu(x):
   return np.maximum(0,x)

def softmax(s):
    expo = np.exp(s)
    expo_sum = np.sum(np.exp(s))
    return expo/expo_sum

def value_between_zero_and_one(val):
    if val > 1:
        val = 1
    if val < 0:
        val = 0
    return val

def show_stats_plot(best_fitness, fastest_finish_times, fastest_car_direction_hist, best_times_alive):
    if len(best_fitness) > 0:
        fig, axs = plt.subplots(4)

        axs[0].plot(range(len(fastest_finish_times)), np.array(fastest_finish_times), label = "Fastest times per generation")
        axs[0].xlabel = "Generation"
        axs[0].ylabel = "fastest_finish_times"

        axs[1].plot(range(len(best_fitness)), best_fitness, label = "Highest fitness per generation")
        axs[1].xlabel = "Generation"
        axs[1].ylabel = "best_fitness"

        axs[2].plot(range(len(best_times_alive)), best_times_alive, label = "best_times_alive")
        axs[2].xlabel = "Generation"
        axs[2].ylabel = "best_times_alive"

        for i in range(len(fastest_car_direction_hist[0])):
            if i == 0: direction = "right"
            if i == 1: direction = "left"
            if i == 2: direction = "brake"
            if i == 3: direction = "acc"
            if i == 4: direction = "forward"
            if i == 5: direction = "glide"
            axs[3].plot(range(len(fastest_car_direction_hist)), np.asarray(fastest_car_direction_hist)[:,i], label = direction)
        axs[3].xlabel = "Direction"
        axs[3].ylabel = "Total times"
        #axs[2].get_yaxis().set_visible(False)
        #(np.array(fastest_finish_times) / 5)
        # axs[2].plot(range(len(best_fitness)), best_fitness)
        # axs[2].plot(range(len(fastest_finish_times)), fastest_finish_times)
        # axs[2].xlabel = "Generation"
        # axs[2].ylabel = "Total pop fit"
        plt.legend()
        plt.show()

def save_fastest_car_to_file(ga, weights_directory):
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)

    filename = weights_directory+'gen=' + str(ga.total_gens) + ' finish_time=' + str(ga.fastest_car.finish_time) + ' car_id=' + str(ga.fastest_car.id)
    outfile = open(filename,'wb')
    pickle.dump(ga.fastest_car.brain,outfile)
    outfile.close()


    # file_name = 'gen=' + str(ga.total_gens) + ' finish_time=' + str(ga.fastest_car.finish_time) + ' car_id=' + str(ga.fastest_car.id)

    # text_file = open(weights_directory+file_name+".txt", "w")

    # text_file.write(np.array2string(ga.fastest_car.brain.weights1.reshape((ga.fastest_car.brain.nr_of_inputs, 10)), separator=',') +" \n next_weight \n")
    # text_file.write(np.array2string(ga.fastest_car.brain.weights2.reshape((10, ga.fastest_car.brain.nr_of_outputs)), separator=','))

    # text_file.close()

def get_saved_car(car, filename):
    # filename = 'jaap_game/weights/best/gen=878 finish_time=1.903 car_id=572207552'
    infile = open(filename,'rb')
    brain = pickle.load(infile)
    infile.close()

    car.brain.weights1 = brain.weights1.reshape(7, 9)
    car.brain.weights2 = brain.weights2.reshape(9, 6)
    car.brain.weights3 = brain.weights3.reshape(6, 2)

    car.car_type = CarType.SAVED
    return car

def check_rect_collision(rect, objects):
        for i in range(len(objects)):
            if rect.colliderect(objects[i]):
                return i
        return -1

def check_car_click_collision(point, objects):
        for i in range(len(objects)):
            if objects[i].rect.collidepoint(point):
                return i
        return -1

def two_point_dist(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angles(A, B, C):
    if (2.0 * A * B) == 0:
        return degrees(acos(int((A * A + B * B - C * C)/(2.01 * A * B))))
    return degrees(acos(int((A * A + B * B - C * C)/(2.0 * A * B))))