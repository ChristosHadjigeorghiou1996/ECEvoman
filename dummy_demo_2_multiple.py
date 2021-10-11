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
experiment_type = ["method_1_random_uniform_initialization", "method_2_smart_initialization"]

experiment_name = "algorithm_b_multiple" 
# experiment_mode = "single"
experiment_mode= "multiple"

# experiment_name = "algorithm_b"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# parameters to tune
hidden_neurons = 10  

# list of enemies 
first_list_of_enemies= [1, 5]
second_list_of_enemies= [2, 4]
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
population_size = 100
# max generations to run
maximum_generations = 30
# total runs to run
total_runs = 10


# max iterations to run without improvement to indicate stagnation
improvement_value = 0
improvement_counter = 0 # cap it to 15
maximum_improvement_counter = 15; # if counter reach break due to stagnation
# parameters for random initialization being between -1 and 1 to be able to change direction
lower_limit_individual_value = -1
upper_limit_individual_value = 1
# trim lower bound to 0 and upper bound to 1 for mutation
probability_lower_bound = 0
probability_upper_bound = 1
# simple straightforward mutation 
mutation_probability=0.05 # paper with hybrid

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

def test_individual(individual, env):
    individual_finess, individual_player_life, individual_enemy_life, individual_run_time = env.play(pcont=individual)
    # apparently creating numpy array from list of lists is depreciated - needs dtype=object
    
    individual_info = np.array((individual_finess, individual_player_life, individual_enemy_life, individual_run_time))
    # print(f'individual_info: {individual_info} and shape: {individual_info.shape}')
    return individual_info

# iterate over the population and estimate the fitness to get the mean
def get_population_information(population, env):
    # list with population_fitness and population individuals 
    combined_list_population_individuals_and_fitness= []
    population_fitness_array = np.zeros(shape=population.shape[0])
    for individual_position in range(population.shape[0]):
        # check if i ever need individual 
        # don't get consider last value for individual playing the game 
        individual_information= test_individual(population[individual_position][:-1], env)
        # print(f'individual_information: {individual_information} [0]: {individual_information[0]}')
        # disregard the return individual as the full array with sigma is used afterwards
        population_fitness_array[individual_position] = individual_information[0]
        combined_list_population_individuals_and_fitness.append([population[individual_position], individual_information[0]]) 
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
    # print(f'combined_list_population_individuals_and_fitness: {combined_list_population_individuals_and_fitness}')
    
    return combined_list_population_individuals_and_fitness, population_fitness_array, most_fit_value, most_fit_individual, mean_fitness_population, standard_deviation_population

# randomly select two parents in the population and combine to produce children per alpha parameter --> random uniform  
def crossover_two_parents_alpha_uniform(first_parent, second_parent):
    # number_of_offspring_pairs= 1
    # print(f'parent_one: {first_parent}\nand shape: {first_parent.shape[0]}')
    # two children
    # number_of_offsprings_array = np.zeros(shape=(number_of_offspring_pairs * 2, first_parent.shape[0]))
    # print(f'number_of_offsprings_array\n{number_of_offsprings_array}\nand shape: {number_of_offsprings_array.shape}')
    # for pair_position in range (number_of_offspring_pairs):
    alpha_parameter= np.random.uniform(probability_lower_bound, probability_upper_bound)        
        # print(f'alpha_parameter: {alpha_parameter}')
    first_offspring= first_parent * alpha_parameter + second_parent * (1-alpha_parameter)
    # sigma of new offspring will be calculated according to their parents 
    # first_offspring[-1] = 1
    # print(f'first_offspring: {first_offspring}\n and shape: {first_offspring.shape}')
    # change the value of sigma for offspring to 1 
    second_offspring= second_parent * alpha_parameter + first_parent * (1-alpha_parameter)
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
        if random_uniform_mutation_probability_array[individual_coordinate_position] <= probability_of_mutation:
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
    
def create_new_population_two_parents_two_offsprings(list_population_values_fitness, old_population, cur_generation, max_generations ):
    # if there are 40 parents they will create 40 offsprings but at the beginning where only cur_generation individuals are 
    # replaced, no need to create that much
    # size_of_mating_population= min(cur_generation * 5, 40)
    size_of_mating_population= min(cur_generation * 10, 60)

    # stochastically create the mating population of k individuals by using sigma scaling and normalization
    # size of mating population is 40 constant 40 but try to make it lower for less than 30 for less generations  
    mating_population= choose_k_individuals_for_mating_stochastically_sigma_scaling(list_population_values_fitness, size_of_mating_population)
    # print(f'mating_population: {mating_population}')
    # offspring array from pairings of mating population
    offspring_population_array= np.zeros(shape=(len(mating_population), len(mating_population[0][0])))
    # print(f'offspring_population_array.shape: {offspring_population_arrcur_generationay.shape}')
        # go over the list pair_wise and select the parents 
    for position_counter in range(0, len(mating_population) - 1, 2):
        # get value of individual parent and disregard fitness
        parent_one= mating_population[position_counter][0]
        # print(f'parent_one: {parent_one}')
        parent_two= mating_population[position_counter + 1][0]
        # print(f'parent_two: {parent_two}')
        offspring_one, offpsring_two= crossover_two_parents_alpha_uniform(parent_one, parent_two)
        mutated_offspring_one= uncorrelated_mutation_with_one_sigma(offspring_one, mutation_probability)
        # print(f'mutated_offspring_one:\n{mutated_offspring_one}\nand shape: {mutated_offspring_one.shape}')
        mutated_offspring_two= uncorrelated_mutation_with_one_sigma(offpsring_two, mutation_probability)
        # print(f'mutated_offspring_one:\n{mutated_offspring_two}\nand shape: {mutated_offspring_two.shape}')
        offspring_population_array[position_counter]= mutated_offspring_one
        offspring_population_array[position_counter + 1]= mutated_offspring_two

    # print(f'offspring_population_array.shape:\n {offspring_population_array.shape}')
    # Assumption that the children will be fitter than the worst individual
    # depending on the generations you want to provide more selection pressure - initially just replace the worst only by one of the offsprings and then replace 
    # all the worst ones. Half way through start replacing more than the worst
    # find worst individual    
    # replace according to the generations, as generations go, selection pressure is increased with the limit of the number of offsprings
    # number_of_individuals_to_replace= min(cur_generation, offspring_population_array.shape[0], 30)
    number_of_individuals_to_replace= min( (max_generations - cur_generation), offspring_population_array.shape[0])

    # print(f'number_of_individuals_to_replace: {number_of_individuals_to_replace} - min( {cur_generation} , {offspring_population_array.shape[0]})')
    positions_of_individual_to_replace= position_of_stochastic_worse_individuals(list_population_values_fitness, number_of_individuals_to_replace)
    for position in range(len(positions_of_individual_to_replace)):
        # print(f'positions_of_individual_to_replace[position]: {positions_of_individual_to_replace[position]}')
        # print(f'check same value from population\n:{old_population[positions_of_individual_to_replace[position]]}')
        # sample through the array but must be list and it is returned in a list which means nead to further access it
        # print(f'make it a list:\n{sample(list(offspring_population_array), k=1)[0]}')
        # print(f'sample(offspring_population_array): {sample(list(offspring_population_array),k=1)[0]}')
        # replace the individuals with sampling from the offsprings
        old_population[positions_of_individual_to_replace[position]] = sample(list(offspring_population_array), k=1)[0]
    # print(f'old_population.shape: {old_population.shape}')      

    # # Randomly select parents from all population
    # # go two steps each time removing two parents and creating two offsprings which are added to the new population. Stop until we are done whichi s of same size.
    # for individual_position in range(0, old_population.shape[0], 2):
    #     # select two parents from the population ensuring they are not the same and forgetting them from the population which assumes they reproduce once 
    #     parent_one, parent_two, old_population= randomly_select_two_individuals(old_population)
    #     # ********** Method 1 - crossover with alpha parameter array to produce 6 children  
    #     # offspring_array= crossover_two_parents_six_children(parent_one, parent_two, cross_over_alpha_parameter_array)

    #     # ********** Method 2 - crossover with alpha parameter uniform - 2 children
    #     first_child, second_child = crossover_two_parents_alpha_uniform(parent_one, parent_two)
    #     # mutate every individual
    #     # mutated_offspring_array= np.zeros(shape=(offspring_array.shape))
    #     # print(f'mutated_offspring_array.shape:\n{mutated_offspring_array.shape}')
    #     # for position, offspring in enumerate(offspring_array): 
    #     #     modified_offspring= non_uniform_mutation_varying_sigma_mutate_individual(offspring, mutation_probability, cur_generation, max_generations)
    #     #     mutated_offspring_array[position]= modified_offspring
    #     first_child_mutated= non_uniform_mutation_varying_sigma_mutate_individual(first_child, mutation_probability, cur_generation, max_generations)
    #     second_child_mutated= non_uniform_mutation_varying_sigma_mutate_individual(second_child, mutation_probability, cur_generation, max_generations)
    #     # works for two offsprings only 
    #     next_generation_population[individual_position]= first_child_mutated[0]
    #     next_generation_population[individual_position + 1]= second_child_mutated[1]
    
    # print(f'next_generation_population: {next_generation_population} and shape: {next_generation_population.shape}')

    return old_population


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
    # to create box plot documentation is followed:  https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html 
    # print(f'current_directory to save box plot outside of runs: {os.getcwd()}')
    # print(f'array received: {array}')
    # print(f'array_received:\n{array}')
    # bo plot in the same plot for each group 
    # box_plot_dict= {}
    # for counter in range(len(enemy_list)): 
    #     box_plot_dict[str(enemy_list[counter])] = array[counter]
    # # print(f'box_plot_dict:\n{box_plot_dict}')
    # fig, ax1 = plt.subplots()
    # ax1.boxplot(box_plot_dict.values())
    # ax1.set_xticklabels(box_plot_dict.keys())

    # fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    # ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    # ax1.set(
    # axisbelow=True,  # Hide the grid behind plot objects
    # title='Mean Gain Over 10 Runs of Best Individual Per Run Per Group ',
    # xlabel='Group Fought Against',
    # ylabel='Fitness',
    # )
    # plt.savefig(f'{algorithm_name}_{len(enemy_list)}_enemies_box_plot.png')
    # plt.close()
    
    # each group has its own box plot 
    for counter in range(len(enemy_list)): 
        box_plot_dict= {}
        box_plot_dict[str(enemy_list[counter])] = array[counter]
        # print(f'box_plot_dict:\n{box_plot_dict}')
        fig, ax1 = plt.subplots()
        ax1.boxplot(box_plot_dict.values())
        ax1.set_xticklabels(box_plot_dict.keys())

        fig.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.25)
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Mean Gain Of 10 Runs of Best Agent Per Group ',
        xlabel='Group Fought Against',
        ylabel='Gain',
        )
        plt.savefig(f'{algorithm_name}_{enemy_list[counter]}_enemies_box_plot.png')
        plt.close()
    
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
    
# define a new cons_multi as required by the docs
def new_cons_multi(values):
    return values.mean()
 
# best individual from both groups which will be compared with dummy_demo multiple 
def get_average_player_life_enemy_life_test_best_individual_five_times_all_enemies_per_run(individual, original_list_of_enemies, env):
    # get a list of all enemies -> 1 .. 8  for 5 runs 
    enemy_list = list(np.arange(8) + 1) 
    best_individual_gain_5_times = np.zeros(shape=(5))

    print('*****Testing five times the best individual of run against all enemies ******************')
    env.update_parameter('enemies', enemy_list)
    for game_runs in range(5):
        individual_information = test_individual(individual[:-1], env)
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
def test_best_individual_five_times_table_per_enemy(individual, env):
    enemy_list = list(np.arange(8) + 1) 
    # print('*****Testing five times the best individual ******************')
    # for each of the 5 games consider the player life and enemy life for each enemy
    best_individual_energy_points= np.zeros(shape=(5, 2 * len(enemy_list)))
    # set multimode off and train per enemy 
    env.update_parameter("multiplemode", "no")
    # now we train per enemy as task 1 
    for pos, enemy in enumerate(enemy_list):
        print(f'enemy: {enemy}')
        env.update_parameter('enemies', enemy)
        for game_runs in range(5):
            individual_information = test_individual(individual[:-1], env)
            print(f'individual_information: {individual_information}')
            best_individual_energy_points[game_runs][pos] = individual_information[1]
            best_individual_energy_points[game_runs][(pos* 2) + 1] = individual_information[2]
    print(f'best_individual_energy_points:\n{best_individual_energy_points}')
    # get the mean values of all 5 runs for each column and use that array for the table 
    mean_best_individual_energy_points_per_enemy= np.zeros(shape=(2 * len(enemy_list)))
    # go through all the columns of player point & enemy point per enemy
    for column in range(best_individual_energy_points.shape[1]):
        array_column_values = best_individual_energy_points[:, column]
        print(f'array_column_values: {array_column_values}')
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
                        print('creating random uniform population')
                        generation_population = create_random_uniform_population(population_size, individuals_size_with_sigma)
                    elif initialization_method == experiment_type[1]:
                        print('creating smart initialized  population')
                        generation_population = create_smart_population(specialized_individuals_population, population_size, individuals_size_with_sigma) 
                else: 
                    generation_population = create_new_population_two_parents_two_offsprings(combined_list_population_values_and_fitness, generation_population, new_generation, maximum_generations)
                # get population information
                combined_list_population_values_and_fitness ,population_fitness_array, best_individual_fitness, best_individual_value, average_fitness_population, standard_deviation_population = get_population_information(generation_population, env) 

                # box plots 
                best_individuals_fitness_populations_array[new_generation] = best_individual_fitness
                best_individuals_value_populations_array[new_generation] = best_individual_value
                # print(f'best_individuals_fitness_populations_array: \n{best_individuals_fitness_populations_array}')
                # print(f'best_individuals_value_populations_array: \n{best_individuals_value_populations_array}')
                # line plot arrays
                max_fitness_ten_runs_twenty_generations_array[current_run][new_generation] = best_individual_fitness
                mean_fitness_ten_runs_twenty_generations_array[current_run][new_generation] = average_fitness_population

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
    finish_time= time.perf_counter()
    print(f'duration: {finish_time - start_time} seconds')