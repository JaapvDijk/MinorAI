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

def sigmoid(s):
        return 1/(1+np.exp(-s))

def relu(s):
   return np.maximum(0,s)

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

def show_stats_plot(best_fitness, fastest_finish_times, fastest_car_direction_hist):
    fig, axs = plt.subplots(3)

    axs[0].plot(range(len(fastest_finish_times)), np.array(fastest_finish_times), label = "Fastest times per generation")
    axs[0].xlabel = "Generation"
    axs[0].ylabel = "Fastest time"

    axs[1].plot(range(len(best_fitness)), best_fitness, label = "Highest fitness per generation")
    axs[1].xlabel = "Generation"
    axs[1].ylabel = "Total pop fit"

    for i in range(len(fastest_car_direction_hist[0])):
        if i == 0: direction = "right"
        if i == 1: direction = "left"
        if i == 2: direction = "nothing/forward"
        axs[2].plot(range(len(fastest_car_direction_hist)), np.asarray(fastest_car_direction_hist)[:,i], label = direction)
    axs[2].xlabel = "Direction"
    axs[2].ylabel = "Total times"
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

    filename = 'gen=' + str(ga.total_gens) + ' finish_time=' + str(ga.fastest_car.finish_time) + ' car_id=' + str(ga.fastest_car.id)
    outfile = open(weights_directory+filename,'wb')
    pickle.dump(ga.fastest_car.brain,outfile)
    outfile.close()


    # file_name = 'gen=' + str(ga.total_gens) + ' finish_time=' + str(ga.fastest_car.finish_time) + ' car_id=' + str(ga.fastest_car.id)

    # text_file = open(weights_directory+file_name+".txt", "w")

    # text_file.write(np.array2string(ga.fastest_car.brain.weights1.reshape((ga.fastest_car.brain.nr_of_inputs, 10)), separator=',') +" \n next_weight \n")
    # text_file.write(np.array2string(ga.fastest_car.brain.weights2.reshape((10, ga.fastest_car.brain.nr_of_outputs)), separator=','))

    # text_file.close()

def get_saved_car_brain(saved_car):
    with open('jaap_game/weights/best/gen=144 time=5.680000000000064 id=446450992.txt', 'r') as file:
        saved_brain_to_use = file.read().replace('\n', '')
    saved_car.brain.weights1 = np.array(eval(saved_brain_to_use.split('next_weight')[0]))
    saved_car.brain.weights2 = np.array(eval(saved_brain_to_use.split('next_weight')[1]))
    return saved_car

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