################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from numpy.core.fromnumeric import shape, size
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
from math import sqrt, exp, floor

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# go again to line and uncomment line 500 in environment 

#seed to have the same pseudorandom variables 
np.random.seed(60)

#smart initialized population, comma instead of plus strategy, self-adaptive memetic with boltzman criteria
# general information 
experiment_type = ["method_1_random_uniform_initialization", "method_2_smart_initialization"]

experiment_name = "algorithm_b_multiple_alg_b" 
# experiment_mode = "single"
experiment_mode= "multiple"

# experiment_name = "algorithm_b"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# parameters to tune
hidden_neurons = 10  

# list of enemies 
first_list_of_enemies= [2, 6]
second_list_of_enemies= [3, 5]
list_of_groups_of_enemies= [ first_list_of_enemies, second_list_of_enemies ]
# assume the same number of enemies in each group
# determines the evaluation function 
enemy_number_per_group = len(first_list_of_enemies)

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
maximum_generations = 30
# total runs to run
total_runs = 2

# max iterations to run without improvement to indicate stagnation
maximum_stagnation_counter = 15; 

# parameters for random initialization being between -1 and 1 to be able to change direction
lower_limit_individual_value = -1
upper_limit_individual_value = 1

# trim lower bound to 0 and upper bound to 1 for mutation
probability_lower_bound = 0
probability_upper_bound = 1

# paper 
mutation_probability=0.05
crossover_probability= 0.80
local_search_probability = 1.0  

# instead of create_random_uniform_population, opt for smart intiialization
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
    # print(f'enemy_one_ten_runs_array_loaded.shape:{enemy_one_ten_runs_array_loaded.shape}')
    # print(f'enemy_one_ten_runs_array_loaded:\n{enemy_one_ten_runs_array_loaded}')
    # print(f'enemy_two_ten_runs_array_loaded:\n{enemy_two_ten_runs_array_loaded}')
    # print(f'enemy_three_ten_runs_array_loaded:\n{enemy_three_ten_runs_array_loaded}')
    # print(f'enemy_four_ten_runs_array_loaded:\n{enemy_four_ten_runs_array_loaded}')
    # print(f'enemy_five_ten_runs_array_loaded:\n{enemy_five_ten_runs_array_loaded}')
    # print(f'enemy_six_ten_runs_array_loaded:\n{enemy_six_ten_runs_array_loaded}')
    # print(f'enemy_seven_ten_runs_array_loaded:\n{enemy_seven_ten_runs_array_loaded}')
    # print(f'enemy_eight_ten_runs_array_loaded:\n{enemy_eight_ten_runs_array_loaded}')
    combined_specialized_individuals= np.vstack((enemy_one_ten_runs_array_loaded,enemy_two_ten_runs_array_loaded, enemy_three_ten_runs_array_loaded, enemy_four_ten_runs_array_loaded, enemy_five_ten_runs_array_loaded, enemy_six_ten_runs_array_loaded, enemy_seven_ten_runs_array_loaded, enemy_eight_ten_runs_array_loaded))
    # print(f'combined_specialized_individuals:\n{combined_specialized_individuals}')
    # print(f'combined_specialized_individuals.shape: {combined_specialized_individuals.shape}')
    # go back to root 
    os.chdir(os.getcwd()+'/..')
    return combined_specialized_individuals
 

def create_smart_population(combined_specialized_population, size_of_population, size_of_individual):
    print('********************Initiating smart population having already loaded specialized individuals **********************')
   # 80 specialized comtrollers so need population_size - 80 individuals
    remaining_population= np.random.uniform(lower_limit_individual_value, upper_limit_individual_value, size=(size_of_population - combined_specialized_population.shape[0], size_of_individual))
    print(f'remaining_population.shape: {remaining_population.shape}')
    print(f'remaining_population:\n{remaining_population}')
    # make sigma of remaining_population to 1 
    remaining_population[:, -1] = 1 
    smart_initialized_population= np.vstack((combined_specialized_population, remaining_population))
    print(f'smart_initialized_population.shape: {smart_initialized_population.shape}')
    print(f'smart_initialized_population:\n{smart_initialized_population}')
    return smart_initialized_population

def create_random_uniform_population(size_of_populations, size_of_individuals):
    new_population = np.random.uniform(lower_limit_individual_value, upper_limit_individual_value, (size_of_populations, size_of_individuals))
    # print(f'new_population: {new_population} and new population.shape: {new_population.shape}')
    # assign the initial sigma as 1 which is the value in the last column of every individual
    new_population[:, -1]= 1
    # print(f'new_population: {new_population} and new population.shape: {new_population.shape}')

    return new_population

def test_individual(individual, env, size_group_of_enemies, multiple_switch):
    # print(F'size_group_of_enemies: {size_group_of_enemies}, multiple_switch: {multiple_switch}')
    individual_finess, individual_player_life, individual_enemy_life, individual_run_time = env.play(pcont=individual)
    # print(f'individual_finess: {individual_finess}, individual_player_life: {individual_player_life} and individual_enemy_life:{individual_enemy_life}')
    # for each group of enemies it returned that many amount of values:

    # MultipleMode:  
    if multiple_switch:
        # Two enemies per group: favour the first enemy over the second: 60% & 40%
        if size_group_of_enemies == 2:  
            # print(f'multiple_switch: {multiple_switch} and size_group_of_enemies: {size_group_of_enemies}')
            individual_info = np.array(( individual_finess[0] * 0.6 + individual_finess[1]*0.4 , individual_player_life[0] * 0.6 + individual_player_life[1]*0.4,
                                        individual_enemy_life[0] * 0.6 + individual_enemy_life[1]*0.4, individual_run_time[0] * 0.6 + individual_run_time[1]*0.4))
        # if there are 8 enemies: favour none -> return the mean 
        else:
            # print(f'multiple_switch: {multiple_switch} and size_group_of_enemies: {size_group_of_enemies}')
            individual_info = np.array(( np.mean(individual_finess), np.mean(individual_player_life), np.mean(individual_enemy_life), np.mean(individual_run_time) ))
    else:
        # print(f'multiple_switch: {multiple_switch}')
        individual_info = np.array((individual_finess, individual_player_life, individual_enemy_life, individual_run_time))
    # print(f'individual_info: {individual_info}')
    return individual_info

# iterate over the population and estimate the fitness to get the mean
def get_population_information(population, env, size_of_group_of_enemies):
    # list with population_fitness and population individuals 
    population_fitness_array = np.zeros(shape=population.shape[0])
    for individual_position in range(population.shape[0]):
        # check if i ever need individual 
        # don't get consider last value for individual playing the game 
        individual_information= test_individual(population[individual_position][:-1], env, size_of_group_of_enemies, multiple_switch=True)
        # print(f'individual_information: {individual_information} [0]: {individual_information[0]}')
        # disregard the return individual as the full array with sigma is used afterwards
        population_fitness_array[individual_position] = individual_information[0]
        # print(f'population_fitness_array:{population_fitness_array}')

    # find the most fit value 
    most_fit_value_position = np.argmax(population_fitness_array)
    most_fit_individual = population[most_fit_value_position]
    most_fit_value = population_fitness_array[most_fit_value_position]
    # print(f'most_fit_value: {most_fit_value}')
    # find average of population 
    mean_fitness_population = np.mean(population_fitness_array)
    # print(f'mean_fitness_population: {mean_fitness_population}')
    # find standard deviation of population
    standard_deviation_population = np.std(population_fitness_array)
    # print(f'standard_deviation_population: {standard_deviation_population}')
    
    return population_fitness_array, most_fit_value, most_fit_individual, most_fit_value_position, mean_fitness_population, standard_deviation_population

# randomly select two parents in the population and combine to produce children per alpha parameter --> random uniform  
def crossover_two_parents_alpha_uniform_with_crossover_probability(first_parent, second_parent, crossover_probability_provided):
    # in this version, the parents will undergo cross over with a given probability # paper 0.8
    specific_crossover_probability = np.random.uniform(probability_lower_bound, probability_upper_bound)
    # print(f'specific_crossover_probability: {specific_crossover_probability}, crossover_probability_provided: {crossover_probability_provided}')
    if specific_crossover_probability <= crossover_probability_provided:
        # print('crossover between parents')
        # number_of_offspring_pairs= 1
        # for pair_position in range (number_of_offspring_pairs):
        alpha_parameter= np.random.uniform(probability_lower_bound, probability_upper_bound)        
            # print(f'alpha_parameter: {alpha_parameter}')
        first_offspring= first_parent * alpha_parameter + second_parent * (1-alpha_parameter)
        # sigma of new offspring will be calculated according to their parents 
        # first_offspring[-1] = 1
        # print(f'first_offspring: {first_offspring}\n and shape: {first_offspring.shape}')
        # change the value of sigma for offspring to 1 
        second_offspring= second_parent * alpha_parameter + first_parent * (1-alpha_parameter)
    else:
        first_offspring = first_parent
        second_offspring = second_parent
    # second_offspring[-1] = 1
    # print(f'second_offspring: {second_offspring}\n and shape: {second_offspring.shape}')

    # print(f'number_of_offsprings_array: {number_of_offsprings_array}')
    return first_offspring, second_offspring

# randomly select two parents in the population and combine to produce two children per alpha parameter --> 6  
# instead of producing two children, produce more lets say 4-6 children and then populate to decide on the fitter version 
# def crossover_two_parents_six_children(first_parent, second_parent, alpha_parameter_array):
#     print(f'alpha_parameter_array:{alpha_parameter_array}')
#     print(f'parent_one: {first_parent}\nand shape: {first_parent.shape[0]}')
#     number_of_offsprings_array = np.zeros(shape=(len(alpha_parameter_array) * 2,first_parent.shape[0]))
#     print(f'number_of_offsprings_array\n{number_of_offsprings_array}\nand shape: {number_of_offsprings_array.shape}')

#     for position, cross_over_paremeter in enumerate(alpha_parameter_array):
#         print(f'position: {position}, crossover_parameter: {cross_over_paremeter}')
#         position_in_array = position * 2
#         # print(f'P1: {first_parent}, P2: {second_parent}')
#         first_offspring = first_parent*cross_over_paremeter + (1-cross_over_paremeter) * second_parent 
#         # print(f'first_offspring: {first_offspring}')
#         # instead of testing now, just give the two offsprings and test once at the end the entire population
#         # first_offspring_values, first_offspring_information = test_individual(first_offspring, env)
#         second_offspring = second_parent * cross_over_paremeter + ( 1 - cross_over_paremeter) * first_parent
#         # print(f'second_offspring: {second_offspring}')
#         # second_offspring_values, second_offspring_information = test_individual(second_offspring, env)
#         # print(f'second_offspring_information: {second_offspring_information}')
#         # return first_offspring_values, first_offspring_information, second_offspring_values, second_offspring_information
#         number_of_offsprings_array[position_in_array] = first_offspring
#         number_of_offsprings_array[position_in_array + 1] = second_offspring
#         print(f'number_of_offsprings_array:{number_of_offsprings_array}')
#     return number_of_offsprings_array

# randomly select two parents and ensure they are not the same from given population
def randomly_select_two_individuals(provided_population):
    # print(f'provided_population.shape: {provided_population.shape}')
    parent_one_position = np.random.randint(0, provided_population.shape[0])
    parent_one= provided_population[parent_one_position]
    # print(f'parent_one_position: {parent_one_position}')
    # mask the value to be excluded
    mask_chosen_value = np.ones(provided_population.shape[0], bool)
    mask_chosen_value[parent_one_position]= False
    parent_two_position = np.random.randint(0, provided_population[mask_chosen_value].shape[0])
    # check if second position is the same or larger than the first --> add one else if it is smaller then alteration 
    # print(f'p2 position before alteration: {parent_two_position}')
    if parent_two_position >= parent_one_position:
        parent_two_position = parent_two_position + 1
    parent_two= provided_population[parent_two_position]
    # print(f'parent_two_position: {parent_two_position} which is: \n {parent_two}')
    # mask second parent
    mask_chosen_value[parent_two_position] = False
    remaining_population= provided_population[mask_chosen_value]
    # print(f'parent_one_position: {parent_one_position} parent_two_position: {parent_two_position}')
    # print(f'remaining_population.shape: {remaining_population.shape}')
    

    return parent_one, parent_two, remaining_population


# cumulative distribution function based on the fitness of each individual of array provided
# Fitness Proportionate Selection and Sigma scaling, # check for min_value ***cite
def choose_k_individuals_for_mating_stochastically_sigma_scaling(received_list, number_of_individuals_to_select ):
    min_value= 0.0001
    fitness_values_list = [row[1] for row in received_list]
    # print(f'fitness_values_list: {fitness_values_list}')
    for position, value in enumerate(fitness_values_list):
        if value <= 0: 
            fitness_values_list[position]= min_value
    # print(f'updated fitness_values_list: {fitness_values_list}')
    x_min= min(fitness_values_list)
    # print(f'x_min: {x_min}')
    x_max = max(fitness_values_list)
    # print(f'x_max: {x_max}')
    
    # sigma scaling --> f'(x) = max( f(x) - (average_f - c * std_f), 0)
    fitness_array= np.asarray(fitness_values_list)
    # print(f'fitness_array: {fitness_array}')
    mean_of_fitness_array= np.mean(fitness_array)
    # print(f'mean_of_fitness_array: {mean_of_fitness_array}')
    standard_deviation_of_fitness_array= np.std(fitness_array)
    # print(f'standard_deviation_of_fitness_array: {standard_deviation_of_fitness_array}')
    # c = 2
    sigma_scaling_array= np.zeros(shape=fitness_array.shape[0])
    for element_position, element_value in enumerate(fitness_array):
        new_value= max(element_value - ( mean_of_fitness_array - 2 * standard_deviation_of_fitness_array), 0)
        sigma_scaling_array[element_position] = new_value
    # print(f'sigma_scaling_array:\n {sigma_scaling_array}')

    # Min Max Normalization from scikit 
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    min_max_scaler= MinMaxScaler()
    # reshaped array row-wise to feed to min_max_scalar
    # normalized values
    # print(f'sigma_scaling_array.reshape(-1, 1):\n {sigma_scaling_array.reshape(-1, 1)}')
    normalized_array= min_max_scaler.fit_transform(sigma_scaling_array.reshape(-1, 1))
    # print(f'normalized_array:\n {normalized_array}')
    normalized_array= np.where(normalized_array==0, min_value,normalized_array)
    # print(f'after replacement normalized_array: {normalized_array}')
    # sum of array for denominator
    sum_normalized= np.sum(normalized_array)
    # print(f'sum_normalized: {sum_normalized}')
    # divide each probability by the total to get cdf 
    normalized_array_divided_by_sum= normalized_array / sum_normalized
    # print(f'normalized_array_divided_by_sum:\n{normalized_array_divided_by_sum}')   
    # print(f'shape: {normalized_array_divided_by_sum.shape}')

    # randomly choose mating pool of parents given the array of parents, the weights and selecting k number with replacement 
    stochastic_choice_of_parents= choices(received_list, weights=normalized_array_divided_by_sum, k=number_of_individuals_to_select)
    # print(f'stochastic_choice_of_parents: {stochastic_choice_of_parents}')
      

    # manual normalization and fps
    # for position, value in enumerate(fitness_values_list):
    #     # normalziation 
    #     new_value= (value - x_min) / (x_max - x_min)
    #     # print(f'new_value: {new_value}')
    #     if new_value == 0:
    #         new_value = min_value
    #     prob_list[position] = new_value
    # # print(f'prob_list: {prob_list}')
    # sum_of_prob_list = sum(prob_list)
    # # print(f'sum_of_prob_list: {sum_of_prob_list}')
    # probability_to_select_parent= [(element / sum_of_prob_list) for element in prob_list]
    # print(f'probability_to_select_parent: {probability_to_select_parent}')
    # # print(sum(probability_to_select_parent))

    # Roulette wheel with cdf regarding the random uniform probability for choosing the parent as long as it was cumulatively more than the drawn probability.
    # Repeat until number of parents is reached 
    # # choose a random parent using uniform prob. [0, 1]
    # parents_chosen_list = []
    # print(f'parents_chosen_list: {parents_chosen_list}')
    # uniform_prob_to_select_parent_array= np.random.uniform(0,1, size=number_of_individuals_to_select)
    # print(f'uniform_prob_to_select_parent_array: {uniform_prob_to_select_parent_array}')
    # for number_of_selected_position in range(uniform_prob_to_select_parent_array.shape[0]):
    #     sum_of_list= 0
    #     for position, value in enumerate(normalized_array_divided_by_sum):
    #         print(f'position: {position}, value: {value}')
    #         sum_of_list = sum_of_list + value 
    #         if sum_of_list >= uniform_prob_to_select_parent_array[number_of_selected_position]:
    #             print(f'sum_of_list: {sum_of_list}')
    #             print(f'position to select: {position}')
    #             parents_chosen_list.append(received_list[position])
    #             break
    # print(f'parents_chosen_list: {parents_chosen_list} and len: {len(parents_chosen_list)}')
    return stochastic_choice_of_parents

# cumulative distribution function based on the fitness of each individual of array but prioritize on worse
# Fitness Proportionate Selection and Sigma scaling
def position_of_stochastic_worse_individuals(received_list, number_of_individuals_to_select ):
    min_value= 0.0001
    fitness_values_list = [row[1] for row in received_list]
    # compute the range of elements in list 
    range_value_list= max(fitness_values_list) - min(fitness_values_list)
    # print(f'range_value_list: {range_value_list}')
    # compute the absolute difference of the value minus the range. This way small fitness values become large and large values small so that you can normalize
    abs_fitness_values_list_minus_range= [(abs(element - range_value_list)) for element in fitness_values_list]
    # print(f'abs_fitness_values_list_minus_range: {abs_fitness_values_list_minus_range}')

    for position, value in enumerate(abs_fitness_values_list_minus_range):
        if value <= 0: 
            abs_fitness_values_list_minus_range[position]= min_value

    # print(f'updated fitness_values_list: {fitness_values_list}')
    x_min= min(abs_fitness_values_list_minus_range)
    # print(f'x_min: {x_min}')
    x_max = max(abs_fitness_values_list_minus_range)
    # print(f'x_max: {x_max}')
    
    # sigma scaling --> f'(x) = max( f(x) - (average_f - c * std_f), 0)
    fitness_array= np.asarray(abs_fitness_values_list_minus_range)
    # print(f'fitness_array: {fitness_array}')
    mean_of_fitness_array= np.mean(abs_fitness_values_list_minus_range)
    # print(f'mean_of_fitness_array: {mean_of_fitness_array}')
    standard_deviation_of_fitness_array= np.std(abs_fitness_values_list_minus_range)
    # print(f'standard_deviation_of_fitness_array: {standard_deviation_of_fitness_array}')
    # c = 2
    sigma_scaling_array= np.zeros(shape=fitness_array.shape[0])
    for element_position, element_value in enumerate(fitness_array):
        new_value= max(element_value - ( mean_of_fitness_array - 2 * standard_deviation_of_fitness_array), 0)
        sigma_scaling_array[element_position] = new_value
    # print(f'sigma_scaling_array:\n {sigma_scaling_array}')

    # Min Max Normalization from scikit 
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    min_max_scaler= MinMaxScaler()
    # reshaped array row-wise to feed to min_max_scalar
    # normalized values
    # print(f'sigma_scaling_array.reshape(-1, 1):\n {sigma_scaling_array.reshape(-1, 1)}')
    normalized_array= min_max_scaler.fit_transform(sigma_scaling_array.reshape(-1, 1))
    # print(f'normalized_array:\n {normalized_array}')

    normalized_array= np.where(normalized_array==0, min_value,normalized_array)
    # print(f'after replacement normalized_array: {normalized_array}')
    # sum of array for denominator
    sum_normalized= np.sum(normalized_array)
    # print(f'sum_normalized: {sum_normalized}')
    # divide each probability by the total to get cdf 
    normalized_array_divided_by_sum= normalized_array / sum_normalized
    # print(f'normalized_array_divided_by_sum:\n{normalized_array_divided_by_sum}')   
    # print(f'shape: {normalized_array_divided_by_sum.shape}')

    # randomly choose mating pool of parents given the array of parents, the weights and selecting k number with 
    index_to_select_list= choices(range(len(received_list)), weights=normalized_array_divided_by_sum, k=number_of_individuals_to_select )
    # print(f'index_to_select_list: {index_to_select_list}')

    return index_to_select_list
    
# log normal mutation operator - logarithm for changing sigma which has normal distribution
def uncorrelated_mutation_with_one_sigma(individual, probability_of_mutation):
    # individual is 265 + 1 
    # each individual coordinate is mutated with a different noise as you draw a new value 
    # value of t is recommended from the literature **cite
    # boundary threshold of too close to 0 then push it to the boundary **cite
    # initial sigma chosen to be 1 
    boundary_threshold = 0.001
    
    learning_rate_t = 1 / (sqrt(individual.shape[0] - 1))
    # get the sigma of the individual which is the last value 
    sigma= individual[individual.shape[0] - 1]
    # compute the new sigma for the individual 
    new_sigma_for_individual = sigma *exp(learning_rate_t * np.random.normal(0, 1))
    # print(f'new_sigma_for_individual  {new_sigma_for_individual} and boundary_threshold {boundary_threshold}')
    if new_sigma_for_individual < boundary_threshold:
        # print(f'new_sigma_for_individual  {new_sigma_for_individual} is less than {boundary_threshold}')
        new_sigma_for_individual = boundary_threshold
        # print(f'new_sigma after boundary threshold {new_sigma_for_individual}')
    # create an np array to pre-compute the probabilities regarding mutation probability per coordinate of individual
    random_uniform_mutation_probability_array= np.random.uniform(probability_lower_bound, probability_upper_bound, individual.shape[0] - 1)
    # create an np array to pre-compute the noise drawn from normal distribution per coordinate
    random_normal_noise_per_coordinate_array= np.random.normal(0, 1, individual.shape[0] - 1)
    
    # last value is sigma 
    for individual_coordinate_position in range(individual.shape[0] - 1):
        # print(f'individual.shape[0] - 1): {individual.shape[0] - 1}')
        # print(f'random_uniform_mutation_probability_array[individual_coordinate_position]: {random_uniform_mutation_probability_array[individual_coordinate_position]} and probability_of_mutation: {probability_of_mutation}')
        if random_uniform_mutation_probability_array[individual_coordinate_position] < probability_of_mutation:
            # print(f'before individual[individual_coordinate_position]: {individual[individual_coordinate_position]}')
            # print(f'random_normal_noise_per_coordinate_array[individual_coordinate_position] : {random_normal_noise_per_coordinate_array[individual_coordinate_position] }')
            new_coordinate_value= individual[individual_coordinate_position] + new_sigma_for_individual *  random_normal_noise_per_coordinate_array[individual_coordinate_position]
            # print(f'individual[individual_coordinate_position]: {individual[individual_coordinate_position]}')
            # print(f'new_coordinate_value: {new_coordinate_value}')
            # ensure is between lower bound and upper bound
            if new_coordinate_value > upper_limit_individual_value:
                new_coordinate_value= upper_limit_individual_value
            elif new_coordinate_value < lower_limit_individual_value:
                new_coordinate_value = lower_limit_individual_value
            individual[individual_coordinate_position] = new_coordinate_value
            # print(f'after individual[individual_coordinate_position]: {individual[individual_coordinate_position]}')

    individual[individual.shape[0] - 1] = new_sigma_for_individual 
    # print(f'individual with new sigma: {individual}')
    return individual

# local search of population 
def get_new_population_from_local_search(old_population_provided, old_population_fit_arr_provided, best_individual_value_position_previous_population_provided, max_fitness_old_gen, mean_fitness_old_gen, env, size_of_enemy_group):
    # pre-initialize the random uniform probability for local search * kinda irrelevant as probability is 1 for local search
    random_uniform_probability_local_search = np.random.uniform(probability_lower_bound, upper_limit_individual_value, size=old_population_provided.shape[0])
   # initialize temperature array which will be inversely proportional to max - average so that it considers the diversity which initially it will be large and then as it converges it will be stricter: > selection pressure
   # feedback from previous generation to determine the temperature --> diversity & selection pressure anchor 
   #  Boltzmann distribution considering the current generation population's temperature 
    temperature= 1 / abs(max_fitness_old_gen - mean_fitness_old_gen)
    # print(f'temperature: {temperature}')
    # rather than checking the position of the best individual vs all others, remove him from the array and add him as it should not be modified
    # delete the row 
    best_individual_previous = old_population_provided[best_individual_value_position_previous_population_provided]
    best_individual_fitness_previous = np.array([ old_population_fit_arr_provided[best_individual_value_position_previous_population_provided] ])
    # print(f'best_individual_previous:\n {best_individual_previous}')
    # print(f'best_individual_fitness_previous: {best_individual_fitness_previous}')
    # print(f'type(best_individual_previous): {type(best_individual_previous)}')
    # print(f'type(best_individual_fitness_previous): {type(best_individual_fitness_previous)}')
    # delete the best and stack it afterwards at the end 
    remaining_population = np.delete(old_population_provided, best_individual_value_position_previous_population_provided, axis=0)
    remaining_population_fitness= np.delete(old_population_fit_arr_provided, best_individual_value_position_previous_population_provided, axis=0)
    # perform local search to each individual 
    for parent_position in range(remaining_population.shape[0]):
        # probability of local search now is set to 1 
        if random_uniform_probability_local_search[parent_position] <= local_search_probability:
            # perform a different move operator than mutation # krasnogor 2002
            # flip the value of a weight in the individual: if positive -> negative, if negative -> positive 
            # use a random number between 1 and instance_size / 10 to choose the number of times this operator will alter the individual: paper
            # don't consider sigma to change it from here 
            max_number_of_bits_to_flip= floor((remaining_population.shape[1] - 1)/ 10)
            # print(f'max_number_of_bits_to_flip: {max_number_of_bits_to_flip}')
            # uniform random choice of number of bits to flip from 1 to above limit
            number_of_bits_to_flip = np.random.randint(1, max_number_of_bits_to_flip)
            # print(f'number_of_bits_to_flip: {number_of_bits_to_flip}')
            # pre-assign the position of the number of the bit to flip in the individual and consider 1 less due to sigma 
            positions_to_flip_in_individual= np.random.randint(0, remaining_population.shape[1]-2, size=number_of_bits_to_flip)
            # print(f'positions_to_flip_in_individual:\n{positions_to_flip_in_individual}')
            # for each position in the individual weight, flip that position's value
            fitness_before= remaining_population_fitness[parent_position]
            # print(f'fitness_before: {fitness_before}')
            # create a new individual which is going to be the same as the other individual before modification if it is not accepted
            modified_individual= remaining_population[parent_position].copy()
            # print(f'modified_individual before:\n{modified_individual}')        
            # get the position of each weight and flip it by multiplying with -1 
            for position in positions_to_flip_in_individual:
                # print(f'position: {position}')
                # print(f'modified_individual[position] before:\n{modified_individual[position]}')
                modified_individual[position] = remaining_population[parent_position][position] * -1 
                # print(f'modified_individual[position] after:\n{modified_individual[position]}')
            # print(f'modified_individual after:\n{modified_individual}')        
            modified_individual_information= test_individual(modified_individual[:-1], env, size_of_enemy_group, multiple_switch=True)
            # print(f'modified_individual_information: {modified_individual_information}')
            # check if the new fitness is better than the old one then accept otherwise depends on probability 
            if modified_individual_information[0] >= fitness_before:
                # print('modified individual is better thus replace')
                # Lamarckian model where individual is replaced by the better version and does not keep genotype
                remaining_population[parent_position]= modified_individual
                remaining_population_fitness[parent_position] = modified_individual_information[0]
                # print(f'remaining_population_fitness after :{remaining_population_fitness[parent_position]}')
                # print(f'remaining_population[parent_position]:\n{remaining_population[parent_position]}')
            else:
                # check the difference
                fitness_difference = fitness_before - modified_individual_information[0]
                # k value is taken from the paper
                k= 0.01 
                threshold_to_replace_with_worse = np.exp( - k * fitness_difference * temperature)
                # print(f'threshold_to_replace_with_worse: {threshold_to_replace_with_worse}')
                # random uniform sample a probability from 0-1 to check if it is less than threshold in which is accepted even tho worse to get out of local optima
                # otherwise just leave the individual as it is in the population
                random_probability_to_replace_with_worse = np.random.uniform(probability_lower_bound, probability_upper_bound)
                # print(f'random_probability_to_replace_with_worse: {random_probability_to_replace_with_worse}')
                if random_probability_to_replace_with_worse <= threshold_to_replace_with_worse:
                    remaining_population[parent_position]= modified_individual
                    remaining_population_fitness[parent_position]= modified_individual_information[0]
                    # print(f'accept worse individual')               

    # print(f'remaining_population.shape:{remaining_population.shape}, best_individual_previous.shape: {best_individual_previous.shape}')
    modified_generation = np.vstack(( remaining_population, best_individual_previous))
    # print(f'modified_generation.shape:{modified_generation.shape}')
    # print(f'remaining_population_fitness.shape:{remaining_population_fitness.shape}, best_individual_fitness_previous.shape: {best_individual_fitness_previous.shape}')
    modified_generation_fitness= np.append(remaining_population_fitness, best_individual_fitness_previous)
    # print(f'modified_generation_fitness.shape:{modified_generation_fitness.shape}, modified_generation_fitness: {modified_generation_fitness}')
    return modified_generation, modified_generation_fitness
    
    # tournament_mode_of_size_two as in the paper
def get_mating_population_via_tournament_mode_with_varying_size(population_array_provided, population_fitness_array_provided, curr_gen_provided, max_gen_provided):
    # mating population is going to be half of the size of population and they will do 2
    # make pairwise comparison and choose the best individual of the two which will be passed to the mating population 
    # for the first rounds use k = 2 and then incfrease k to increase selection pressure
    # up to generation and including gen_16, k =2 then up to gen_24: k =3 and the remaining k=4
    # random choice of size of k, check literature 
    # deterministic tournament instead of pT   
    size_of_k = max(2,  int(np.ceil(curr_gen_provided/8)))
    # print(f'tournament size_of_k: {size_of_k} ')
    # allow duplication of individuals for k 
    # create a numbered list of the index of the population array
    numbered_list_of_individuals = list(np.arange(population_array_provided.shape[0]))
    # print(f'population_array_provided.shape: {population_array_provided.shape}')
    mating_population= np.zeros(shape=(population_array_provided.shape))
    # print(f'mating_population.shape: {mating_population.shape}')
    for index in range(mating_population.shape[0]):
        # uniform select the individuals to participate in the tournamentsticky 
        index_list_of_participants= choices(numbered_list_of_individuals, k= size_of_k)
        # print(f'index_list_of_participants: {index_list_of_participants}') 
        # for each index in the population get the fitness of each individual
        fitness_list = [ population_fitness_array_provided[x] for x in index_list_of_participants]
        # print('fitness_list\n:', fitness_list)
        # find the max fitness position 
        largest_value_position= np.argmax(fitness_list)
        # print('largest_value_position:', largest_value_position, 'and value: ', population_fitness_array_provided[index_list_of_participants[largest_value_position]])
        # get the individual and add them to the mating population
        mating_population[index] = population_array_provided[index_list_of_participants[largest_value_position]]
    return mating_population 

    # main function to perform evolution on current generation's population
def perform_evolution(old_population, old_population_fitness_array, cur_generation, max_generations, best_individual_value_position_previous_population, max_fitness_old_generation, mean_fitness_old_generation, env, size_enemy_group ):
    # this version will change the population (μ, λ) which is lowest selection pressure  all individuals are replaced except the best  
    population_after_local_search_array, population_fitness_after_local_search_array= get_new_population_from_local_search(old_population, old_population_fitness_array, best_individual_value_position_previous_population, max_fitness_old_generation, mean_fitness_old_generation, env, size_enemy_group)
    # tournament selection of size 2 
    mating_selection = get_mating_population_via_tournament_mode_with_varying_size(population_after_local_search_array, population_fitness_after_local_search_array, cur_generation, max_generations) 
    
    # print(f'mating_selection: {mating_selection}')
    # offspring array from pairings of mating population
    offspring_population_array= np.zeros(shape=mating_selection.shape)
    # print(f'offspring_population_array.shape: {offspring_population_arrcur_generationay.shape}')
        # go over the list pair_wise and select the parents 
    for position_counter in range(0, len(mating_selection), 2):
        # print(f'position_counter: {position_counter}')
        # get value of individual parent and disregard fitness
        parent_one= mating_selection[position_counter]
        # print(f'parent_one: {parent_one}')
        parent_two= mating_selection[position_counter + 1]
        # print(f'parent_two: {parent_two}')
        offspring_one, offpsring_two= crossover_two_parents_alpha_uniform_with_crossover_probability(parent_one, parent_two, crossover_probability)
        mutated_offspring_one= uncorrelated_mutation_with_one_sigma(offspring_one, mutation_probability)
        # print(f'mutated_offspring_one:\n{mutated_offspring_one}\nand shape: {mutated_offspring_one.shape}')
        mutated_offspring_two= uncorrelated_mutation_with_one_sigma(offpsring_two, mutation_probability)
        # print(f'mutated_offspring_one:\n{mutated_offspring_two}\nand shape: {mutated_offspring_two.shape}')
        offspring_population_array[position_counter]= mutated_offspring_one
        offspring_population_array[position_counter + 1]= mutated_offspring_two

    return offspring_population_array


# stagnation for max value check - if for 12 generations it is not improved replaced 20% of the population at random to inject a diversity / spread boost 
def stagnation_escape_function( population_provided):
    # continue from here
    print('***************************STAGNATION******************************')
    number_of_individuals_to_replace = population_provided.shape[0] / 5
    index_list_of_individuals_to_replace = list(np.arange(population_provided.shape[0]))
    # sample without repeatition each individual's index to be replaced 
    index_individual_to_replace = sample(index_list_of_individuals_to_replace, number_of_individuals_to_replace)
    # preinitialize the array of random uniform individuals to replace them faster
    individuals_to_replace_with= np.random.uniform(lower_limit_individual_value, upper_limit_individual_value, size=(number_of_individuals_to_replace, individuals_size_with_sigma)) 
    # change their sigma to 1 
    individuals_to_replace_with[:, -1] = 1    
    # print(f'index_individual_to_replace:\n{index_individual_to_replace}')
    # go over the positions of index to replace and for each index get the position in the original array to replace with the index of the randomly created individual  
    for index_individual_to_remove in range(len(index_individual_to_replace)):
        # print(f'index_individual_to_remove: {index_individual_to_remove} which is in index_individual_to_replace: {index_individual_to_replace[index_individual_to_remove]} ')
        # print(f'Replaced with: {individuals_to_replace_with[index_individual_to_remove]}')
        population_provided[index_individual_to_replace[index_individual_to_remove]] = individuals_to_replace_with[index_individual_to_remove]
    # print(f'afterwards, population_provided:\n{population_provided}')
    
    return population_provided

# # write numpy arrays to files 
# def save_array_to_files_with_defined_parameters(experiment_name_folder, initial_directory_to_load_enemy, enemy_list, current_enemy_index, number_of_runs, current_run_value, number_of_generations, number_of_individuals, best_individuals_fitness_per_population, best_individual_per_population_array, average_fitness_per_population, standard_deviation_per_population):
#     # get time to attach it to the folder 
#     from datetime import datetime
#     if os.getcwd() == initial_directory_to_load_enemy:
#         folder_name = (f"{datetime.now().strftime('%d_%m__%Y_%H_%M_%S_')}{len(enemy_list)}_enemies_{number_of_runs}number_of_runs_{number_of_generations}_generations_{number_of_individuals}_number_of_individuals")
#         print(f'changing folder name to {folder_name}')
#     # # create and navigate to the experiment_name folder 
#     if not os.path.exists(experiment_name_folder):
#         os.makedirs(experiment_name_folder)
#     os.chdir(os.getcwd() +'/'+experiment_name_folder)
#     # print(f'working directory: {os.getcwd()}')
#     # create folder with time and specific tuple 
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#     os.chdir(os.getcwd() + '/'+ folder_name)
#     # create a folder for enemies
#     if current_enemy_index != len(enemy_list):
#         enemy_path_name= {str(enemy_list[current_enemy_index]) + '_enemy'}
#         if not os.path.exists(enemy_path_name):
#             os.mkdir(enemy_path_name)
#     os.chdir(os.getcwd() + '/'+ enemy_path_name)
#     # create a fodler for runs
#     if current_run_value != (number_of_runs):
#         os.mkdir(str(current_run_value)+'_run')
#         os.chdir(os.getcwd()+'/'+str(current_run_value)+'_run')
#     print(f'current directory to save the arrays: {os.getcwd()}')
#     # save the numpy arrays individually 
#     np.save('best_individuals_fitness_per_population', best_individuals_fitness_per_population)
#     np.save('best_individual_per_population_array', best_individual_per_population_array)
#     np.save('average_fitness_per_population', average_fitness_per_population)
#     np.save('standard_deviation_per_population', standard_deviation_per_population)
#     os.chdir('../../../../')
#     # print(f'final_directory after save: {os.getcwd()}')
#     # print(f'check folder_name: {folder_name}')
    # return folder_name

# def load_numpy_files(first_algorithm_folder, specific_name_of_folder, specific_run):
#     print(os.getcwd())
#     os.chdir(os.getcwd() +'/' +first_algorithm_folder+'/' +specific_name_of_folder+'/'+specific_run+'_run')

#     print(f'new_directory: {os.getcwd()}')
#     # os.chdir(path_till_numpy_files)
#     average_fitness_per_population_array_name = "average_fitness_per_population.npy"
#     average_fitness_per_population_array = np.load(average_fitness_per_population_array_name)
#     print(f'average_fitness_per_population_array:\n{average_fitness_per_population_array}')
#     best_individuals_fitness_per_population_array_name = "best_individuals_fitness_per_population.npy"
#     best_individuals_fitness_per_population_array = np.load(best_individuals_fitness_per_population_array_name)
#     print(f'best_individuals_fitness_per_population_array:\n{best_individuals_fitness_per_population_array}')
#     # array with the best individual per population
#     best_individual_per_population_array_name = "best_individual_per_population_array.npy"
#     best_individual_per_population_array = np.load(best_individual_per_population_array_name)
#     print(f'best_individual_per_population_array:\n{best_individual_per_population_array}')
#     # array of standard_deviations of all populations 
#     standard_deviation_per_population_array_name = "standard_deviation_per_population.npy"
#     standard_deviation_per_population_array = np.load(standard_deviation_per_population_array_name)
#     print(f'standard_deviation_per_population_array:\n{standard_deviation_per_population_array}')
#     return average_fitness_per_population_array, best_individuals_fitness_per_population_array, best_individual_per_population_array, standard_deviation_per_population_array

# def create_graphs_for_each_run(average_fitness_per_population_array, best_individuals_fitness_per_population_array, standard_deviation_per_population, algorithm_name, working_directory, run_number):
#     print(f'current_working_directory for graphs: {os.getcwd()}')
#     #go to algorithm folder, specific date and in each run_number 
#     os.chdir(algorithm_name+'/'+working_directory+'/'+str(run_number)+'_run')
#     print(f'now in: {os.getcwd()}')
#     # plot average fitness per population
#     plt.plot(average_fitness_per_population_array)
#     plt.xlabel('Generations')
#     plt.ylabel('Fitness Value')
#     plt.title('Average fitness per population')
#     plt.savefig(f'{run_number}_average_fitness_per_population.png')
#     plt.close()
#     # plot best_individual_fitness_per_population
#     plt.plot(best_individuals_fitness_per_population_array, 'ro')
#     plt.xlabel('Generations')
#     plt.ylabel('Fitness Value')
#     plt.title('Fittest individual per population')
#     plt.axis([0, len(best_individuals_fitness_per_population_array), min(best_individuals_fitness_per_population_array) - 5, max(best_individuals_fitness_per_population_array) + 5])
#     plt.savefig(f'{run_number}_fittest_individual_per_population.png')
#     plt.close()
#     # plot standard deviation per population
#     plt.plot(standard_deviation_per_population)
#     plt.xlabel('Generations')
#     plt.ylabel('Fitness Value')
#     plt.title('Standard Deviation per population')
#     plt.savefig(f'{run_number}_standard_deviation_per_population.png')
#     plt.close()
#     os.chdir('../../../')
    # print(f'check_directory: {os.getcwd()}')



# write numpy arrays to files 
def create_directory_to_save_graphs(experiment_name_folder, initial_directory_to_load_enemy, enemy_list, number_of_runs, number_of_generations, number_of_individuals, box_plot_array, line_plot_avg_arr, line_plot_max_array, line_plot_std_max_array, line_plot_std_avg_arr, enemy_groups_best_individual_values, enemy_groups_best_individual_fitness):
    # box plot - 1 
    # print('box_plot_array\n', box_plot_array)
    # line plot - 4 
    # print('line_plot_avg_arr\n', line_plot_avg_arr)
    # print('line_plot_max_array\n', line_plot_max_array)
    # print('line_plot_std_max_array\n', line_plot_std_max_array)
    # print('line_plot_std_avg_arr\n', line_plot_std_avg_arr)
 
    # get time to attach it to the folder 
    from datetime import datetime
    if os.getcwd() == initial_directory_to_load_enemy:
        folder_name = (f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S_')}{len(enemy_list)}_enemies_{number_of_runs}_number_of_runs_{number_of_generations}_generations_{number_of_individuals}_number_of_individuals")
        # print(f'changing folder name to {folder_name}')
    # # create and navigate to the experiment_name folder 
    if not os.path.exists(experiment_name_folder):
        os.makedirs(experiment_name_folder)
    os.chdir(os.getcwd() +'/'+experiment_name_folder)
    # print(f'working directory: {os.getcwd()}')
    # create folder with time and specific tuple 
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(os.getcwd() + '/'+ folder_name)
    current_folder= os.getcwd()
    # print(f'folder saving arrays: {os.getcwd()}')
    # boxplot 
    np.save('box_plot_array.npy', box_plot_array)
    # line graph
    np.save('line_plot_avg_arr.npy', line_plot_avg_arr)
    np.save('line_plot_max_array.npy', line_plot_max_array)
    np.save('line_plot_std_max_array.npy', line_plot_std_max_array)
    np.save('line_plot_std_avg_arr.npy', line_plot_std_avg_arr)
    # get the other algorithm best groups and load both algorithms --> 4 best individual as 2 algorithms with each 2 groups
    np.save('enemy_groups_best_individual_value.npy', enemy_groups_best_individual_values)
    np.save('enemy_groups_best_individual_fitness.npy', enemy_groups_best_individual_fitness)

    return current_folder
    
def visualize_box_plot(array, algorithm_name, enemy_list ):

    # bo plot in the same plot for each group 
    box_plot_dict= {}
    for counter in range(len(enemy_list)): 
        box_plot_dict[str(enemy_list[counter])] = array[counter]
    # print(f'box_plot_dict:\n{box_plot_dict}')
    fig, ax1 = plt.subplots()
    ax1.boxplot(box_plot_dict.values())
    ax1.set_xticklabels(box_plot_dict.keys())
    fig.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.25)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    ax1.set(
    axisbelow=True,  # Hide the grid behind plot objects
    title='Mean Gain Of 10 Runs of Best Agent Of 5 Runs',
    xlabel='Group Trained Against',
    ylabel='Gain',
    )
    plt.savefig(f'{algorithm_name}_{enemy_list[0]}_{enemy_list[1]}_enemies_box_plot.png')
    plt.close()
    
    #     # each group has its own box plot 
    # for counter in range(len(enemy_list)): 
    #     box_plot_dict= {}
    #     box_plot_dict[str(enemy_list[counter])] = array[counter]
    #     # print(f'box_plot_dict:\n{box_plot_dict}')
    #     fig, ax1 = plt.subplots()
    #     ax1.boxplot(box_plot_dict.values())
    #     ax1.set_xticklabels(box_plot_dict.keys())

    #     fig.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.25)
    #     ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    #     ax1.set(
    #     axisbelow=True,  # Hide the grid behind plot objects
    #     title='Mean Gain Of 10 Runs of Best Agent Of 5 Runs',
    #     xlabel='Group Trained Against',
    #     ylabel='Gain',
    #     )
    #     plt.savefig(f'{algorithm_name}_{enemy_list[counter]}_enemies_box_plot.png')
    #     plt.close()
    
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

    # have all line plots in one graph
    # _, ax1 = plt.subplots()
    # ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    # ax1.set(
    # axisbelow=True,  # Hide the grid behind plot objects
    # title="Average / Max Fitness Per Generation Per Group of Enemy",
    # xlabel='Generations',
    # ylabel='Fitness',
    # )
    # number_of_generations_array=  list(np.arange(average_fitness_all_runs_per_generation.shape[1]))
    # # print(f'number_of_generations_array:\n{number_of_generations_array}')
    # linestyle_plot= ["dashed", "solid", "dotted" ]
    # colors_plot= ['b', 'g', 'r']
    # for counter in range(len(enemies_list)):
    #     plt.errorbar(number_of_generations_array, max_fitness_all_runs_per_generation[counter], yerr= standard_deviation_max_fitness_all_runs_per_generation[counter] ,color=colors_plot[counter], ecolor=colors_plot[counter], linestyle=linestyle_plot[counter], label=(f"Max Fit. En. {enemies_list[counter]}"))
    #     plt.errorbar(number_of_generations_array, average_fitness_all_runs_per_generation[counter], yerr= standard_deviation_average_fitness_all_runs_per_generation[counter] ,color=colors_plot[counter], ecolor=colors_plot[counter], linestyle=linestyle_plot[counter], label=f"Avg. Fit. En. {enemies_list[counter]}", )
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot#:~:text=To%20place%20the%20legend%20outside,left%20corner%20of%20the%20legend.&text=A%20more%20versatile%20approach%20is,placed%2C%20using%20the%20bbox_to_anchor%20argument. - 
    # # Put a legend below current axis
    # # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    # plt.subplots_adjust(right=0.7)
    # plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    # plt.savefig(f'{algorithm_name}_line_plot.png')
    # # plt.show()
    # plt.close()

# write down the best result for the competition:
def write_the_best_individual_out_of_both_groups(both_groups_best_individuals_values, both_groups_best_individuals_fitness, list_of_groups, env):
    folder_to_come_back_to = os.getcwd()
    print(f'folder_to_come_back_to: {folder_to_come_back_to}')
    os.chdir('../../')
    print(f'now in: {os.getcwd()}')
    # list of groups is the list of two groups of x enemies so get the length of the first index 
    size_of_group_of_enemies= len(list_of_groups[0])
    # print(f'both_groups_best_individuals_values:\n{both_groups_best_individuals_values}\n and both_groups_best_individuals_fitness:{both_groups_best_individuals_fitness}')
    position_of_highest= np.argmax(both_groups_best_individuals_fitness)
    best_fitness_of_experiment= both_groups_best_individuals_fitness[position_of_highest]
    best_individual_of_experiment= both_groups_best_individuals_values[position_of_highest]
    print(f'Highest Fitness: {best_fitness_of_experiment} from:\n{best_individual_of_experiment}')
    mean_of_individual_against_all_enemies = test_best_individual_five_times_table_per_enemy(best_individual_of_experiment, size_of_group_of_enemies, env)
    print(f'mean_of_individual_against_all_enemies:\n{mean_of_individual_against_all_enemies}')
    # now go back to the folder 
    os.chdir(folder_to_come_back_to)
    print(f'now in: {os.getcwd()}')
    # saves file with the best solution as required by the file system
    np.savetxt('best.txt',best_individual_of_experiment)
    np.savetxt(f'{list_of_groups}_table.txt', mean_of_individual_against_all_enemies)
    print('finished writing the file')
    
    
# define a new cons_multi as required by the docs
def new_cons_multi(values):
    # return values.mean()
    return values 
 
# best individual from both groups which will be compared with dummy_demo multiple 
def get_average_player_life_enemy_life_test_best_individual_five_times_all_enemies_per_run(individual, original_list_of_enemies, env):
    # get a list of all enemies -> 1 .. 8  for 5 runs 
    enemy_list = list(np.arange(8) + 1) 
    best_individual_gain_5_times = np.zeros(shape=(5))

    print('*****Testing five times the best individual of run against all enemies ******************')
    env.update_parameter('enemies', enemy_list)
    for game_runs in range(5):
        individual_information = test_individual(individual[:-1], env, len(enemy_list), multiple_switch=True)
        best_individual_gain_5_times[game_runs]= individual_information[1] - individual_information[2]
    print(f'best_individual_gain_5_times:\n{best_individual_gain_5_times}')
    mean_best_individual_gain= np.mean(best_individual_gain_5_times)
    # save the best individual for competition
    # np.save(f'best_individual.npy', individual)
    print(f'mean_best_individual_gain:\n{mean_best_individual_gain}')
    # set the list of enemies back to the original 
    env.update_parameter('enemies', original_list_of_enemies)
    return mean_best_individual_gain



# this is for the table which is called for all the enemies 
# overall enemies not for each enemy *********************************************
def test_best_individual_five_times_table_per_enemy(individual, size_of_each_group_of_enemies, env):
    enemy_list = list(np.arange(8) + 1) 
    # print('*****Testing five times the best individual ******************')
    # for each of the 5 games consider the player life and enemy life for each enemy
    best_individual_energy_points= np.zeros(shape=(5, 2 * len(enemy_list)))
    # set multimode off and train per enemy 
    env.update_parameter("multiplemode", "no")
    # now we train per enemy as task 1 
    for pos, enemy in enumerate(enemy_list):
        env.update_parameter('enemies', [enemy])
        print(f'pos: {pos} and enemy: {enemy}')
        for game_runs in range(5):
            individual_information = test_individual(individual[:-1], env, size_of_each_group_of_enemies, multiple_switch=False)
            # print(f'individual_information: {individual_information}')
            best_individual_energy_points[game_runs][pos * 2] = individual_information[1]
            best_individual_energy_points[game_runs][(pos* 2) + 1] = individual_information[2]
    #     print(f'best_individual_energy_points:\n{best_individual_energy_points}')
    # print(f'final best_individual_energy_points:\n{best_individual_energy_points}')
    # get the mean values of all 5 runs for each column and use that array for the table 
    mean_best_individual_energy_points_per_enemy= np.zeros(shape=(2 * len(enemy_list)))
    # go through all the columns of player point & enemy point per enemy
    for column in range(best_individual_energy_points.shape[1]):
        array_column_values = best_individual_energy_points[:, column]
        # print(f'array_column_values: {array_column_values}')
        mean_best_individual_energy_points_per_enemy[column]= np.mean(array_column_values)  

    return mean_best_individual_energy_points_per_enemy


if __name__ == '__main__':
    # start the timer
    start_time= time.perf_counter()
    # load the specialized individuals once in case they are going to be used via the smart initialization. If not used, simply ignore it. 
    specialized_individuals_population= load_specialized_individuals_per_enemy()

    # one individual for each group for the values and for the fitness
    groups_array_best_individual_value = np.zeros(shape=(len(list_of_groups_of_enemies), individuals_size_with_sigma))
    groups_array_best_fitness = np.zeros(shape=(len(list_of_groups_of_enemies)))
    # make an array for each enemy fpr the box plot 
    different_groups_ten_runs_five_times_individual_gains_arrays= np.zeros(shape=(len(list_of_groups_of_enemies), total_runs))
    # print(f'different_groups_ten_runs_five_times_individual_gains_arrays.shape: {different_groups_ten_runs_five_times_individual_gains_arrays.shape}')

    # get average fitness and standarddeviation for all runs per generation per enemy 
    different_groups_average_fitness_all_runs_per_generation= np.zeros(shape=(len(list_of_groups_of_enemies), maximum_generations))
    different_groups_standard_deviation_average_fitness_all_runs_per_generation= np.zeros(shape=(len(list_of_groups_of_enemies), maximum_generations))
    # get max fitness and standard deviation for all runs per generation per enemy
    different_groups_max_fitness_all_runs_per_generation= np.zeros(shape= (len(list_of_groups_of_enemies), maximum_generations))
    different_groups_standard_deviation_max_fitness_all_runs_per_generation= np.zeros(shape=(len(list_of_groups_of_enemies), maximum_generations))
    
    for group_index in range(len(list_of_groups_of_enemies)):
        # get the list of enemies to have the individual play against 
        list_of_enemies= list_of_groups_of_enemies[group_index]
        print(f'group: {group_index} with enemies: {list_of_enemies}')
        env.update_parameter('enemies', list_of_enemies)
        env.cons_multi = new_cons_multi
        initial_directory_to_load_enemy= os.getcwd()    

        if os.getcwd() != initial_directory_to_load_enemy:
            os.chdir('../../')
        # print(f'current_working_directory: {os.getcwd()}')


        # Line plot array for each generation per run - have run in rows and generations in columns and then pick up columnwise via [:, generation_number]
        max_fitness_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, maximum_generations)) 
        mean_fitness_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, maximum_generations))

        # get the best individual of all generations per run per enemy and its fitness
        ten_runs_best_individuals_arrays = np.zeros(shape=(total_runs, individuals_size_with_sigma ))
        ten_runs_best_individuals_fitness_arrays = np.zeros(shape=(total_runs))

        # ******* Box plot array 
        for current_run in range(total_runs):
            # best individual at start of run to check for stagnation 
            best_individual_fitness_value_stagnation = - 10
            stagnation_current_value = 0
            # once the counter goes to the max, turn the switch to true and next generation instead of evolving it will undergo stagnation_escape_function
            stagnation_function_next_generation_switch = False 
            print(f'******************************Starting run {current_run} / {total_runs - 1}')
            # keep records of the best individual fitness, best individual, mean and sd of each population
            best_individuals_fitness_populations_array = np.zeros(shape=(maximum_generations))
            best_individuals_value_populations_array = np.zeros(shape=(maximum_generations, individuals_size_with_sigma))
            for new_generation in range(maximum_generations):
                # print(f'********************Starting generation {new_generation}/{maximum_generations-1} **************************************')
                # add stagnation 
                # if maximum_improvement_counter != improvement_counter:
                if new_generation == 0:
                    # flip the initialization_method for experiment type to change EA initialization method
                    # experiment_type[0] is the random initialization, [1] is the smart initialization
                    initialization_method= experiment_type[0]
                    if initialization_method== experiment_type[0]:
                        generation_population = create_random_uniform_population(population_size, individuals_size_with_sigma)
                    elif initialization_method == experiment_type[1]:
                        generation_population = create_smart_population(specialized_individuals_population, population_size, individuals_size_with_sigma) 
                else:
                    if stagnation_function_next_generation_switch == False: 
                        generation_population = perform_evolution(generation_population, population_fitness_array, new_generation, maximum_generations, best_individual_value_position, best_individual_fitness, average_fitness_population, env, len(list_of_enemies))
                    else: 
                        # check if stagnation turns to true --> instead of evoluting, randomly swap 20 individuals 
                        generation_population = stagnation_escape_function(generation_population)
                        # flip the stagnation switch and reset counter and stagnation_best
                        stagnation_function_next_generation_switch = False
                        stagnation_current_value= 0
                        best_individual_fitness_value_stagnation = - 10
                # get population information
                population_fitness_array, best_individual_fitness, best_individual_value, best_individual_value_position, average_fitness_population, standard_deviation_population = get_population_information(generation_population, env, len(list_of_enemies)) 

                # keep a score for best individual in each generation - this will trigger stagnation_escape if for 12 gens it is not improved

                # box plots 
                best_individuals_fitness_populations_array[new_generation] = best_individual_fitness
                best_individuals_value_populations_array[new_generation] = best_individual_value
                # line plot arrays
                max_fitness_ten_runs_twenty_generations_array[current_run][new_generation] = best_individual_fitness
                mean_fitness_ten_runs_twenty_generations_array[current_run][new_generation] = average_fitness_population

                # check if the max value for stagnation is worse than the individual of current generation and update the value and set counter to zero otherwise increment counter
                if best_individual_fitness_value_stagnation < best_individual_fitness:
                    # update stagnation fitness value 
                    best_individual_fitness_value_stagnation = best_individual_fitness
                    # reset stagnation counter
                    stagnation_current_value = 0 
                else: 
                    # increment the stagnation value and check if it reached cap for 
                    stagnation_current_value = stagnation_current_value + 1 
                    if stagnation_current_value == maximum_stagnation_counter:
                        print('****Reach stagnation limit, call function to randomize percentage *************')
                        stagnation_function_next_generation_switch = True                     

            # print("*******************************************************STATISTICS*****************************")

            # Box Plot Data
                # find best individual of all generations
            # print(f'best_individuals_fitness_populations_array:\n{best_individuals_fitness_populations_array}')
            best_individual_fitness_position_all_generations = np.argmax(best_individuals_fitness_populations_array)
            # print(f'\nbest_individual_fitness_position_all_generations:\n{best_individual_fitness_position_all_generations}')
            best_individual_value_all_generations= best_individuals_value_populations_array[best_individual_fitness_position_all_generations]
            # get the fitness value of that individual 
            best_individual_fitness_all_generations= best_individuals_fitness_populations_array[best_individual_fitness_position_all_generations]
            # print(f'\nbest_individual_value_all_generations:\n{best_individual_value_all_generations}')

            # this is per run, once populated, save it  
            ten_runs_best_individuals_arrays[current_run]= best_individual_value_all_generations
            ten_runs_best_individuals_fitness_arrays[current_run] = best_individual_fitness_all_generations
            

            # # check the current run being total_runs - 1 and then save to get the values of best individual of all runs per enemy 
            # if current_run == (total_runs - 1):
            #     print(f'ten_runs_best_individuals_arrays:\n{ten_runs_best_individuals_arrays}')
            #     np.save(f'enemy_{list_of_enemies[enemy_index]}_ten_runs_best_individuals_arrays.npy', ten_runs_best_individuals_arrays)

            

            # test that individual 5 times and get the mean of those 5 
            mean_of_five_times_gains_best_individual_array= get_average_player_life_enemy_life_test_best_individual_five_times_all_enemies_per_run(best_individual_value_all_generations, list_of_enemies, env)
                # get the mean of the five_times_scores
            # mean_of_five_times_scores_best_individual = np.mean(five_times_scores_best_individual_array)
            # print(f'mean_of_five_times_scores_best_individual: {mean_of_five_times_scores_best_individual}')

            different_groups_ten_runs_five_times_individual_gains_arrays[group_index][current_run] = mean_of_five_times_gains_best_individual_array
            # print(f'different_groups_ten_runs_five_times_individual_gains_arrays[{group_index}][{current_run}]: {different_groups_ten_runs_five_times_individual_gains_arrays}')
        
        # Once all runs have finished find the best individual per enemy and get the value, fitness and gains 
        position_of_best_individual_fitness_per_enemy_for_all_runs_array = np.argmax(ten_runs_best_individuals_fitness_arrays)
        groups_array_best_individual_value[group_index] = ten_runs_best_individuals_arrays[position_of_best_individual_fitness_per_enemy_for_all_runs_array]
        groups_array_best_fitness[group_index] = ten_runs_best_individuals_fitness_arrays[position_of_best_individual_fitness_per_enemy_for_all_runs_array]   
        env.update_solutions([groups_array_best_individual_value[group_index], groups_array_best_fitness[group_index]])
        env.save_state()    

        #******* Line plot Average and max 
        for generation_counter in range(maximum_generations):
            # average fitness 
            current_generation_average_fitness= mean_fitness_ten_runs_twenty_generations_array[:, generation_counter]
            # print(f'current_generation_average_fitness: {current_generation_average_fitness}')
            different_groups_average_fitness_all_runs_per_generation[group_index][generation_counter]= np.mean(current_generation_average_fitness)
            different_groups_standard_deviation_average_fitness_all_runs_per_generation[group_index][generation_counter]= np.std(current_generation_average_fitness)
            # max fitness 
            current_generation_max_fitness= max_fitness_ten_runs_twenty_generations_array[:, generation_counter]
            # print(f'current_generation_max_fitness: {current_generation_max_fitness}')
            different_groups_max_fitness_all_runs_per_generation[group_index][generation_counter]= np.max(current_generation_max_fitness)
            different_groups_standard_deviation_max_fitness_all_runs_per_generation[group_index][generation_counter]= np.std(current_generation_max_fitness)


    working_directory= create_directory_to_save_graphs(experiment_name, initial_directory_to_load_enemy, list_of_groups_of_enemies, total_runs, maximum_generations, population_size, different_groups_ten_runs_five_times_individual_gains_arrays, different_groups_average_fitness_all_runs_per_generation, different_groups_max_fitness_all_runs_per_generation, different_groups_standard_deviation_max_fitness_all_runs_per_generation, different_groups_standard_deviation_average_fitness_all_runs_per_generation, groups_array_best_individual_value, groups_array_best_fitness )
    visualize_box_plot(different_groups_ten_runs_five_times_individual_gains_arrays, experiment_name, list_of_groups_of_enemies)
    # Draw Line Plot and save in current folder  FOR ALL ENEMIES
    draw_line_plot(different_groups_average_fitness_all_runs_per_generation, different_groups_max_fitness_all_runs_per_generation, different_groups_standard_deviation_max_fitness_all_runs_per_generation, different_groups_standard_deviation_average_fitness_all_runs_per_generation, experiment_name, list_of_groups_of_enemies)
    # find out the best individual of the two groups and write him in txt 
    write_the_best_individual_out_of_both_groups(groups_array_best_individual_value, groups_array_best_fitness, list_of_groups_of_enemies, env)

    finish_time= time.perf_counter()
    print(f'duration: {finish_time - start_time} seconds')