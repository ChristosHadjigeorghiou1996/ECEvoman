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
# sqrt and exp for the uncorrelated mutation with one sigma
from math import sqrt, exp

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# go again to line and uncomment line 500 in environment 

# general information 
experiment_type = "test"
# experiment_type = "test_b"
experiment_name = "algorithm_b_multiple" 
# experiment_mode = "single"
experiment_mode= "multiple"

# experiment_name = "algorithm_b"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# parameters to tune
hidden_neurons = 10  

# list of enemies 
first_list_of_enemies= [2, 8]
second_list_of_enemies= [3, 5]
list_of_groups_of_enemies= [ first_list_of_enemies, second_list_of_enemies ]

env = Environment(
                experiment_name=experiment_name, 
                enemies= list_of_groups_of_enemies[0],
                multiplemode= "yes",
                playermode='ai',
                # playermode="human",
                randomini= "yes",
                # playermode="human",
                # loadenemy="no",
                player_controller=player_controller(hidden_neurons),
                speed="fastest",
                enemymode='static',
                level=2)

# if we are using uncorrelated mutation with one sigma -> 266 
individuals_size_with_sigma = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5 + 1 
# 

# population size made of n individuals
population_size = 10
# max generations to run
maximum_generations = 3
# total runs to run
total_runs = 3

def draw_line_plot(average_fitness_all_runs_per_generation, max_fitness_all_runs_per_generation, standard_deviation_average_fitness_all_runs_per_generation, standard_deviation_max_fitness_all_runs_per_generation, algorithm_name, enemies_list):
    # print(f'average_fitness_all_runs_per_generation.shape:\n{average_fitness_all_runs_per_generation.shape}')
    # print(f'standard_deviation_average_fitness_all_runs_per_generation.shape:\n{standard_deviation_average_fitness_all_runs_per_generation.shape}')
    # print(f'max_fitness_all_runs_per_generation.shape:\n{max_fitness_all_runs_per_generation.shape}')
    # print(f'standard_deviation_max_fitness_all_runs_per_generation.shape:\n{standard_deviation_max_fitness_all_runs_per_generation.shape}')


    for counter in range(len(enemies_list)):
        _, ax1 = plt.subplots()
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title="Average / Max Fitness Per Generation Per Group of Enemy",
        xlabel='Generations',
        ylabel='Fitness',
        )
        number_of_generations_array=  list(np.arange(average_fitness_all_runs_per_generation.shape[1]))
        # print(f'number_of_generations_array:\n{number_of_generations_array}')
        linestyle_plot= ["dashed", "solid", "dotted" ]
        colors_plot= ['b', 'g', 'r']
        plt.errorbar(number_of_generations_array, max_fitness_all_runs_per_generation[counter], yerr= standard_deviation_max_fitness_all_runs_per_generation[counter] ,color=colors_plot[counter], ecolor=colors_plot[counter], linestyle=linestyle_plot[counter], label=(f"Max Fit. En. {enemies_list[counter]}"))
        plt.errorbar(number_of_generations_array, average_fitness_all_runs_per_generation[counter], yerr= standard_deviation_average_fitness_all_runs_per_generation[counter] ,color=colors_plot[counter], ecolor=colors_plot[counter], linestyle=linestyle_plot[counter], label=f"Avg. Fit. En. {enemies_list[counter]}", )
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot#:~:text=To%20place%20the%20legend%20outside,left%20corner%20of%20the%20legend.&text=A%20more%20versatile%20approach%20is,placed%2C%20using%20the%20bbox_to_anchor%20argument. - 
        # Put a legend below current axis
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        plt.subplots_adjust(right=0.7)
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.savefig(f'{algorithm_name}_{enemies_list[counter]}__line_plot.png')
        # plt.show()
        plt.close()



def load_numpy_files_and_draw_line_graph():
    print(f'current_directory: {os.getcwd()}')

    # print(f'new_directory: {os.getcwd()}')
    # os.chdir(path_till_numpy_files)
    folder_to_go= "algorithm_b_multiple/08_10_2021_16_46_43_2_enemies_3_number_of_runs_3_generations_10_number_of_individuals"
    os.chdir(os.getcwd() + "/"+ folder_to_go)
    line_plot_avg_arr = "line_plot_avg_arr.npy"
    line_plot_max_array = "line_plot_max_array.npy"
    line_plot_std_avg_arr = "line_plot_std_avg_arr.npy"
    line_plot_std_max_array = "line_plot_std_max_array.npy"
    line_plot_avg_arr_loaded = np.load(line_plot_avg_arr)
    line_plot_max_array_loaded = np.load(line_plot_max_array)
    line_plot_std_avg_arr_loaded = np.load(line_plot_std_avg_arr)
    line_plot_std_max_array_loaded = np.load(line_plot_std_max_array)
    draw_line_plot(line_plot_avg_arr_loaded, line_plot_max_array_loaded, line_plot_std_max_array_loaded, line_plot_std_avg_arr_loaded, experiment_name, list_of_groups_of_enemies)

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

def create_smart_population(combined_specialized_population, size_of_population, size_of_individual):
   # 80 specialized comtrollers so need population_size - 80 individuals
    remaining_population= np.random.uniform(-1, 1, size=(size_of_population - combined_specialized_population.shape[0], size_of_individual))
    print(f'remaining_population.shape: {remaining_population.shape}')
    print(f'remaining_population:\n{remaining_population}')
    # make sigma of remaining_population to 1 
    remaining_population[:, -1] = 1 
    smart_initialized_population= np.vstack((combined_specialized_population, remaining_population))
    print(f'smart_initialized_population.shape: {smart_initialized_population.shape}')
    print(f'smart_initialized_population:\n{smart_initialized_population}')
    return smart_initialized_population

# load the specialized_individuals once and only the remaining random in each time 
def load_specialized_individuals_per_enemy():
    print(f'current_directory: {os.getcwd()}')
    # print(f'new_directory: {os.getcwd()}')
    # os.chdir(path_till_numpy_files)
    folder_to_go= "specialized_agents"
    os.chdir(os.getcwd() + "/"+ folder_to_go)
    enemy_one_ten_runs_array = "enemy_1_ten_runs_best_individuals_arrays.npy"
    enemy_one_ten_runs_array_loaded = np.load(enemy_one_ten_runs_array)
    enemy_two_ten_runs_array = "enemy_2_ten_runs_best_individuals_arrays.npy"
    enemy_two_ten_runs_array_loaded = np.load(enemy_two_ten_runs_array)
    enemy_three_ten_runs_array = "enemy_3_ten_runs_best_individuals_arrays.npy"
    enemy_three_ten_runs_array_loaded = np.load(enemy_three_ten_runs_array)
    enemy_four_ten_runs_array = "enemy_4_ten_runs_best_individuals_arrays.npy"
    enemy_four_ten_runs_array_loaded = np.load(enemy_four_ten_runs_array)
    enemy_five_ten_runs_array = "enemy_5_ten_runs_best_individuals_arrays.npy"
    enemy_five_ten_runs_array_loaded = np.load(enemy_five_ten_runs_array)
    enemy_six_ten_runs_array = "enemy_6_ten_runs_best_individuals_arrays.npy"
    enemy_six_ten_runs_array_loaded = np.load(enemy_six_ten_runs_array)
    enemy_seven_ten_runs_array = "enemy_7_ten_runs_best_individuals_arrays.npy"
    enemy_seven_ten_runs_array_loaded = np.load(enemy_seven_ten_runs_array)
    enemy_eight_ten_runs_array = "enemy_8_ten_runs_best_individuals_arrays.npy"
    enemy_eight_ten_runs_array_loaded = np.load(enemy_eight_ten_runs_array)
    print(f'enemy_one_ten_runs_array_loaded.shape:{enemy_one_ten_runs_array_loaded.shape}')
    print(f'enemy_one_ten_runs_array_loaded:\n{enemy_one_ten_runs_array_loaded}')
    print(f'enemy_two_ten_runs_array_loaded:\n{enemy_two_ten_runs_array_loaded}')
    print(f'enemy_three_ten_runs_array_loaded:\n{enemy_three_ten_runs_array_loaded}')
    print(f'enemy_four_ten_runs_array_loaded:\n{enemy_four_ten_runs_array_loaded}')
    print(f'enemy_five_ten_runs_array_loaded:\n{enemy_five_ten_runs_array_loaded}')
    print(f'enemy_six_ten_runs_array_loaded:\n{enemy_six_ten_runs_array_loaded}')
    print(f'enemy_seven_ten_runs_array_loaded:\n{enemy_seven_ten_runs_array_loaded}')
    print(f'enemy_eight_ten_runs_array_loaded:\n{enemy_eight_ten_runs_array_loaded}')
    combined_specialized_individuals= np.vstack((enemy_one_ten_runs_array_loaded,enemy_two_ten_runs_array_loaded, enemy_three_ten_runs_array_loaded, enemy_four_ten_runs_array_loaded, enemy_five_ten_runs_array_loaded, enemy_six_ten_runs_array_loaded, enemy_seven_ten_runs_array_loaded, enemy_eight_ten_runs_array_loaded))
    print(f'combined_specialized_individuals:\n{combined_specialized_individuals}')
    print(f'combined_specialized_individuals.shape: {combined_specialized_individuals.shape}')
    return combined_specialized_individuals


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

# enemy_8_example = load_numpy_files()
# print(f'enemy_1_file.shape:\n {enemy_8_example.shape}')
# # test the first best individual 
# first_run_individual = enemy_8_example[0]
# five_times_scores_best_individual_array= test_best_individual_five_times(first_run_individual, env)
# load_numpy_files_and_draw_line_graph()
# load specialized agents folder 
if __name__ == "__main__":
    specialized_population= load_specialized_individuals_per_enemy()
    final_population = create_smart_population(specialized_population, 100, 266)
    print(f'final_population:\n{final_population}')
    print(f'final_population.shape:\n{final_population.shape}')
