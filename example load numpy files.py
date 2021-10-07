################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from numpy.core.fromnumeric import shape
from numpy.lib import load
from numpy.lib.function_base import average
sys.path.insert(0, 'evoman') 
from demo_controller import player_controller
from environment import Environment
# import numpy for random initialization and arrays
import numpy as np 
# import matplot to visualize arrays and draw graphs
import matplotlib.pyplot as plt
# min max scalar for normalization instead of manual method
from sklearn.preprocessing import MinMaxScaler
# random.choices allows stochastic mating population for choose_k_individuals_for_mating_stochastically_sigma_scaling function 
from random import choices, sample
# time will enable timing of the algorithm
import time

# general infromation 
experiment_type = "test"
experiment_name = "algorithm_a_test_three_enemies" 
experiment_mode = "single"
working_directory= "not_yet"
# experiment_mode="multiple"
# experiment_name = "algorithm_b"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# parameters to tune
hidden_neurons = 10  
 
if experiment_mode == "single":
    list_of_enemies = [8] # 5 to show
    multiple= "no"
    speed_switch="fastest"
    # speed_switch="normal"
    
if experiment_type == "test":
    env = Environment(
                    experiment_name=experiment_name, 
                    enemies=[list_of_enemies[0]],
                    multiplemode= multiple,
                    playermode='ai',
                    # playermode="human",
                    randomini= "yes",
                    # playermode="human",
                    # loadenemy="no",
                    player_controller=player_controller(hidden_neurons),
                    speed=speed_switch,
                    enemymode='static')

def load_numpy_files():
    print(f'current_directory: {os.getcwd()}')

    # print(f'new_directory: {os.getcwd()}')
    # os.chdir(path_till_numpy_files)
    enemy_8_ten_runs_best_individuals_arrays = "enemy_8_ten_runs_best_individuals_arrays.npy"
    enemy_8_ten_runs_best_individuals_arrays_loaded = np.load(enemy_8_ten_runs_best_individuals_arrays)
    # # print(f'average_fitness_per_population_array:\n{average_fitness_per_population_array}')
    # best_individuals_fitness_per_population_array_name = "best_individuals_fitness_per_population.npy"
    # best_individuals_fitness_per_population_array = np.load(best_individuals_fitness_per_population_array_name)
    # # print(f'best_individuals_fitness_per_population_array:\n{best_individuals_fitness_per_population_array}')
    # # array with the best individual per population
    # best_individual_per_population_array_name = "best_individual_per_population_array.npy"
    # best_individual_per_population_array = np.load(best_individual_per_population_array_name)
    # # print(f'best_individual_per_population_array:\n{best_individual_per_population_array}')
    # # array of standard_deviations of all populations 
    # standard_deviation_per_population_array_name = "standard_deviation_per_population.npy"
    # standard_deviation_per_population_array = np.load(standard_deviation_per_population_array_name)
    # # print(f'standard_deviation_per_population_array:\n{standard_deviation_per_population_array}')
    return enemy_8_ten_runs_best_individuals_arrays_loaded

def test_best_individual_five_times(individual, env):
    # print('*****Testing five times the best individual ******************')
    best_individual_scores= np.zeros(shape=5)
    for game_runs in range(5):
        _, individual_information = test_individual(individual[:-1], env)
        best_individual_scores[game_runs] = individual_information[0]
    print(f'best_individual_score:\n{best_individual_scores}')
    return best_individual_scores

def test_individual(individual, env):
    individual_finess, individual_player_life, individual_enemy_life, individual_run_time = env.play(pcont=individual)
    # apparently creating numpy array from list of lists is depreciated - needs dtype=object
    
    individual_info = np.array((individual_finess, individual_player_life, individual_enemy_life, individual_run_time))
    # print(f'individual_info: {individual_info} and shape: {individual_info.shape}')
    return individual, individual_info

enemy_8_example = load_numpy_files()
print(f'enemy_8_file:\n {enemy_8_example}')
# test the first best individual 
first_run_individual = enemy_8_example[0]
five_times_scores_best_individual_array= test_best_individual_five_times(first_run_individual, env)
