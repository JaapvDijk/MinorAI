import numpy as np

def sigmoid(s):
        return 1/(1+np.exp(-s))

def relu(s):
   return np.maximum(0,s)

def softmax(s):
    expo = np.exp(s)
    expo_sum = np.sum(np.exp(s))
    return expo/expo_sum

def show_stats_plot():
    fig, axs = plt.subplots(3)

    axs[0].plot(range(len(fastest_finish_times)), fastest_finish_times, label='Fastest finish time')
    axs[0].xlabel = "Generation"
    axs[0].ylabel = "Fastest time"

    axs[1].plot(range(len(best_fitness)), best_fitness, label='Fitness of best car')
    axs[1].xlabel = "Generation"
    axs[1].ylabel = "Total pop fit"

    axs[2].plot(range(len(best_fitness)), best_fitness, label='Fitness of best car')
    axs[2].plot(range(len(fastest_finish_times)), fastest_finish_times, label='Fastest finish time')
    axs[2].xlabel = "Generation"
    axs[2].ylabel = "Tot fit vs fastest time"

    plt.legend()
    plt.show()

def save_fastest_car_to_file(ga):
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)

    file_name = 'gen=' + str(total_gens) + ' finish_time=' + str(ga.fastest_car.finish_time) + ' car_id=' + str(ga.fastest_car.id)

    text_file = open(weights_directory+file_name+".txt", "w")

    text_file.write(np.array2string(ga.fastest_car.brain.weights1.reshape((ga.fastest_car.brain.nr_of_inputs, 20)), separator=',') +" \n next_weight \n")
    text_file.write(np.array2string(ga.fastest_car.brain.weights2.reshape((20, ga.fastest_car.brain.nr_of_outputs)), separator=','))

    text_file.close()


def get_saved_car_brain():
    brain = NeuralNetwork()

    with open('jaap_game/weights/best/gen=144 time=5.680000000000064 id=446450992.txt', 'r') as file:
        saved_brain_to_use = file.read().replace('\n', '')
    brain.weights1 = np.array(eval(saved_brain_to_use.split('next_weight')[0]))
    brain.weights2 = np.array(eval(saved_brain_to_use.split('next_weight')[1]))
    return brain

def check_rect_collision(rect, objects):
        for i in range(len(objects)):
            if rect.colliderect(objects[i]):
                return i
        return -1