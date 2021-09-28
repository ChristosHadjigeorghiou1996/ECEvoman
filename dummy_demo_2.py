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
# import math for floor and sqrt of learning rate t
from math import sqrt, exp

# general infromation 
experiment_type = "test"
# experiment_type = "test_b"
experiment_name = "algorithm_a_test" 
experiment_mode = "single"
working_directory= "not_yet"
# experiment_mode="multiple"
# experiment_name = "algorithm_b"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# parameters to tune
hidden_neurons = 10  

# 
if experiment_mode == "single":
    list_of_enemies = [2] # 5 to show
    multiple= "no"
    print('single')
    speed_switch="fastest"
    # speed_switch="normal"
    # stotsi taxitita

elif experiment_mode == "multiple":
    list_of_enemies = [1, 3, 5]
    multiple = "yes"
    speed_switch="normal"
    print('multiple')

if experiment_type == "test":
    env = Environment(
                    experiment_name=experiment_name, 
                    enemies=list_of_enemies,
                    multiplemode= multiple,
                    playermode='ai',
                    # playermode="human",
                    randomini= "yes",
                    # playermode="human",
                    # loadenemy="no",
                    player_controller=player_controller(hidden_neurons),
                    speed=speed_switch,
                    enemymode='static')
elif experiment_type == "":
    print('eleos')

# size of individuals # 265 
# individuals_size = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5
# if we are using uncorrelated mutation with one sigma -> 266 
individuals_size_with_sigma = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5 + 1 

# population size made of n individuals
population_size = 10
# max generations to run
maximum_generations = 5
# total runs to run
total_runs = 3 

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
mutation_probability=0.25 # random number check literature
# consider self-adaptin mutation 
# crossover alpha for arithmetic crossover & later blend 
crossover_alpha_parameter= 0.4 

# # deap 
# from deap import base, creator, tools
# creator.create("FitnessMax", base.fitness, weights=((population_size,individuals_size)))
# creator.create("Individual", np.array, fitness=creator.FitnessMax)

# toolbox = base.Toolbox()
# toolbox.register("attribute", np.random.uniform(lower_bound, upper_bound, individuals_size))
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=population_size)
# toolbox.register("population", tools.initRepeat, np.ndarray, toolbox.individual)    


## simple genetics
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
    return individual, individual_info

# iterate over the population and estimate the fitness to get the mean
def get_population_information(population, env):
    # list with population_fitness and population individuals 
    combined_list_population_individuals_and_fitness= []
    population_fitness_array = np.zeros(shape=population.shape[0])
    for individual_position in range(population.shape[0]):
        # check if i ever need individual 
        # don't get consider last value for individual playing the game 
        _, individual_information= test_individual(population[individual_position][:-1], env)
        # print(f'individual_information: {individual_information} [0]: {individual_information[0]}')
        # disregard the return individual as the full array with sigma is used afterwards
        population_fitness_array[individual_position] = individual_information[0]
        combined_list_population_individuals_and_fitness.append([population[individual_position], individual_information[0]]) 
        # print(f'population_fitness_array:{population_fitness_array}')

    # find the most fit value 
    most_fit_value_position = np.argmax(population_fitness_array)
    most_fit_individual = population[most_fit_value_position]
    most_fit_value = population_fitness_array[most_fit_value_position]
    print(f'most_fit_value: {most_fit_value}')
    # find average of population 
    mean_fitness_population = np.mean(population_fitness_array)
    print(f'mean_fitness_population: {mean_fitness_population}')
    # find standard deviation of population
    standard_deviation_population = np.std(population_fitness_array)
    print(f'standard_deviation_population: {standard_deviation_population}')
    # print(f'combined_list_population_individuals_and_fitness: {combined_list_population_individuals_and_fitness}')
    
    return combined_list_population_individuals_and_fitness, population_fitness_array, most_fit_value, most_fit_individual, mean_fitness_population, standard_deviation_population

# randomly select two parents in the population and combine to produce two children per alpha parameter --> 6  
# instead of producing two children, produce more lets say 4-6 children and then populate to decide on the fitter version 
def crossover_two_parents_six_children(first_parent, second_parent, alpha_parameter_array):
    print(f'alpha_parameter_array:{alpha_parameter_array}')
    print(f'parent_one: {first_parent}\nand shape: {first_parent.shape[0]}')
    number_of_offsprings_array = np.zeros(shape=(len(alpha_parameter_array) * 2,first_parent.shape[0]))
    print(f'number_of_offsprings_array\n{number_of_offsprings_array}\nand shape: {number_of_offsprings_array.shape}')

    for position, cross_over_paremeter in enumerate(alpha_parameter_array):
        print(f'position: {position}, crossover_parameter: {cross_over_paremeter}')
        position_in_array = position * 2
        # print(f'P1: {first_parent}, P2: {second_parent}')
        first_offspring = first_parent*cross_over_paremeter + (1-cross_over_paremeter) * second_parent 
        # print(f'first_offspring: {first_offspring}')
        # instead of testing now, just give the two offsprings and test once at the end the entire population
        # first_offspring_values, first_offspring_information = test_individual(first_offspring, env)
        second_offspring = second_parent * cross_over_paremeter + ( 1 - cross_over_paremeter) * first_parent
        # print(f'second_offspring: {second_offspring}')
        # second_offspring_values, second_offspring_information = test_individual(second_offspring, env)
        # print(f'second_offspring_information: {second_offspring_information}')
        # return first_offspring_values, first_offspring_information, second_offspring_values, second_offspring_information
        number_of_offsprings_array[position_in_array] = first_offspring
        number_of_offsprings_array[position_in_array + 1] = second_offspring
        print(f'number_of_offsprings_array:{number_of_offsprings_array}')
    return number_of_offsprings_array

# randomly select two parents in the population and combine to produce children per alpha parameter --> random uniform  
def crossover_two_parents_alpha_uniform(first_parent, second_parent):
    number_of_offspring_pairs= 1
    # print(f'parent_one: {first_parent}\nand shape: {first_parent.shape[0]}')
    # two children
    number_of_offsprings_array = np.zeros(shape=(number_of_offspring_pairs * 2, first_parent.shape[0]))
    # print(f'number_of_offsprings_array\n{number_of_offsprings_array}\nand shape: {number_of_offsprings_array.shape}')
    for pair_position in range (number_of_offspring_pairs):
        alpha_parameter= np.random.uniform(probability_lower_bound, probability_upper_bound)        
        # print(f'alpha_parameter: {alpha_parameter}')
        first_offspring= first_parent * alpha_parameter + second_parent * (1-alpha_parameter)
        number_of_offsprings_array[pair_position]=first_offspring
        second_offspring= second_parent * alpha_parameter + first_parent * (1-alpha_parameter)
        number_of_offsprings_array[pair_position+1]= second_offspring
    # they start with sigma parameter 1 
    number_of_offsprings_array[:, -1] = 1
    # print(f'number_of_offsprings_array: {number_of_offsprings_array}')
    return number_of_offsprings_array


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

# rank based selection to sort population and give probabilities according to fitness 
def rank_based_selection(lista_population_value_and_fitness):
    sorted_list = sorted(lista_population_value_and_fitness, key=lambda fitness: fitness[1], reverse=True)
    # print(f'sorted_list: {sorted_list[:3]}')
    # x_new = (x_i - x_min) / x_max - x_min 
    check_less_zero= [(x <= 0) for x in sorted_list[1]]
    return sorted_list
    
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
    


    


    
    previous_sigma= individual[len(individual.shape[0])]

def non_uniform_mutation_varying_sigma_mutate_individual(individual, current_generation, total_number_of_generations):
    # non-uniform mutation 
    # the idea is to start with a high sigma value and as generations proceed to reduce the step --> sigma = 1 - (current/ max)
    # current generations will be 1 instead of 0 as 0 is creation of uniform population thus -1 for the sake of starting with 1 as sigma
    varying_sigma_value = 1 - ((current_generation - 1) / total_number_of_generations)
    # print(f'varying_sigma: {varying_sigma_value}')
    # create the random probabilities for the individual from before
    random_uniform_mutation_probability_array= np.random.uniform(lower_bound, upper_bound, individual.shape[0])
    # print(f'random_uniform_mutation_probability_array: {random_uniform_mutation_probability_array}')
    # print(f'random_probability: {random_probability}')
    # if prob > 0.25 
    for individual_position, individual_value in enumerate(individual):
        if random_uniform_mutation_probability_array[individual_position] > mutation_probability:
            # print(f'random_uniform_mutation_probability_array[individual_position]: {random_uniform_mutation_probability_array[individual_position]}')
            # print(f'individual_position: {individual_position} and value: {individual_value}')
            # mutate by considering some noise drawn from Gaussian distribution and then check if the random probability is > 0.5 in which case add otherwise subtract 
            mutated_value = individual_value + np.random.normal(0, varying_sigma_value)
            # print(f'mutated_value before: {mutated_value}')
            if mutated_value < lower_limit_individual_value:
                mutated_value = lower_limit_individual_value
            elif mutated_value > upper_limit_individual_value:
                mutated_value = upper_limit_individual_value
            # print(f'mutated_value after: {mutated_value}')
            individual[individual_position] = mutated_value

            # print(f'mutated individual: {mutated_individual}')
    return individual
        # maybe consider uniform from -1 to 1 instead 
    
def create_new_population_two_parents_two_offsprings(list_population_values_fitness, old_population, old_population_fitness, cur_generation, max_generations):
    # cross over alpha parameter varying to create more offsprings. sharing most / a lot of information in each child 
    # cross_over_alpha_parameter_array= np.array([0.1, 0.25, 0.4])
    # print(f'old_population.shape: {old_population.shape} and values: {old_population}')

    # same size population is maintained 
    next_generation_population=np.zeros_like(old_population)
    # print(f'next_generation_population.shape: {next_generation_population.shape} and values: {next_generation_population}')
    

    # rank_based_population = rank_based_selection(list_population_values_fitness )
    # print('de panw')


    # Randomly select parents 
    # go two steps each time removing two parents and creating two offsprings which are added to the new population. Stop until we are done whichi s of same size.
    for individual_position in range(0, old_population.shape[0], 2):
        # select two parents from the population ensuring they are not the same and forgetting them from the population which assumes they reproduce once 
        parent_one, parent_two, old_population= randomly_select_two_individuals(old_population)
        # ********** Method 1 - crossover with alpha parameter array to produce 6 children  
        # offspring_array= crossover_two_parents_six_children(parent_one, parent_two, cross_over_alpha_parameter_array)

        # ********** Method 2 - crossover with alpha parameter uniform - 2 children
        offspring_array= crossover_two_parents_alpha_uniform(parent_one, parent_two)
        # mutate every individual
        mutated_offspring_array= np.zeros(shape=(offspring_array.shape))
        # print(f'mutated_offspring_array.shape:\n{mutated_offspring_array.shape}')
        for position, offspring in enumerate(offspring_array): 
            # modified_offspring= non_uniform_mutation_varying_sigma_mutate_individual(offspring, cur_generation, max_generations)
            # check uncorrelated_mutation_with_one_sigma
            modified_offspring= uncorrelated_mutation_with_one_sigma(offspring, mutation_probability)
            mutated_offspring_array[position]= modified_offspring
        # works for two offsprings only 
        next_generation_population[individual_position]= mutated_offspring_array[0]
        next_generation_population[individual_position + 1]= mutated_offspring_array[1]
    
    # print(f'next_generation_population: {next_generation_population} and shape: {next_generation_population.shape}')

    return next_generation_population


# mutation --> random < 0.5 then modify sigma

# write numpy arrays to files 
def save_array_to_files_with_defined_parameters(experiment_name_folder, folder_name, number_of_runs, current_run_value, number_of_generations, number_of_individuals, best_individuals_fitness_per_population, best_individual_per_population_array, average_fitness_per_population, standard_deviation_per_population):
    # get time to attach it to the folder 
    from datetime import datetime
    if folder_name == "not_yet":
        folder_name = (f"{datetime.now().strftime('%H_%M_%S_')}{number_of_runs}number_of_runs_{number_of_generations}_generations_{number_of_individuals}_number_of_individuals")
        print(f'changing folder name to {folder_name}')
    # # create and navigate to the experiment_name folder 
    if not os.path.exists(experiment_name_folder):
        os.makedirs(experiment_name_folder)
    os.chdir(os.getcwd() +'/'+experiment_name_folder)
    print(f'working directory: {os.getcwd()}')
    # create folder with time and specific tuple 
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(os.getcwd() + '/'+ folder_name)
    if current_run_value != (number_of_runs):
        os.mkdir(str(current_run_value)+'_run')
        os.chdir(os.getcwd()+'/'+str(current_run_value)+'_run')
    print(f'current directory to save the arrays: {os.getcwd()}')
    # save the numpy arrays individually 
    np.save('best_individuals_fitness_per_population', best_individuals_fitness_per_population)
    np.save('best_individual_per_population_array', best_individual_per_population_array)
    np.save('average_fitness_per_population', average_fitness_per_population)
    np.save('standard_deviation_per_population', standard_deviation_per_population)
    os.chdir('../../../')
    print(f'final_directory after save: {os.getcwd()}')
    print(f'check folder_name: {folder_name}')
    return folder_name

def load_numpy_files(first_algorithm_folder, specific_name_of_folder, specific_run):
    print(os.getcwd())
    os.chdir(os.getcwd() +'/' +first_algorithm_folder+'/' +specific_name_of_folder+'/'+specific_run+'_run')

    print(f'new_directory: {os.getcwd()}')
    # os.chdir(path_till_numpy_files)
    average_fitness_per_population_array_name = "average_fitness_per_population.npy"
    average_fitness_per_population_array = np.load(average_fitness_per_population_array_name)
    print(f'average_fitness_per_population_array:\n{average_fitness_per_population_array}')
    best_individuals_fitness_per_population_array_name = "best_individuals_fitness_per_population.npy"
    best_individuals_fitness_per_population_array = np.load(best_individuals_fitness_per_population_array_name)
    print(f'best_individuals_fitness_per_population_array:\n{best_individuals_fitness_per_population_array}')
    # array with the best individual per population
    best_individual_per_population_array_name = "best_individual_per_population_array.npy"
    best_individual_per_population_array = np.load(best_individual_per_population_array_name)
    print(f'best_individual_per_population_array:\n{best_individual_per_population_array}')
    # array of standard_deviations of all populations 
    standard_deviation_per_population_array_name = "standard_deviation_per_population.npy"
    standard_deviation_per_population_array = np.load(standard_deviation_per_population_array_name)
    print(f'standard_deviation_per_population_array:\n{standard_deviation_per_population_array}')
    return average_fitness_per_population_array, best_individuals_fitness_per_population_array, best_individual_per_population_array, standard_deviation_per_population_array

def create_graphs_for_each_run(average_fitness_per_population_array, best_individuals_fitness_per_population_array, standard_deviation_per_population, algorithm_name, working_directory, run_number):
    print(f'current_working_directory for graphs: {os.getcwd()}')
    #go to algorithm folder, specific date and in each run_number 
    os.chdir(algorithm_name+'/'+working_directory+'/'+str(run_number)+'_run')
    print(f'now in: {os.getcwd()}')
    # plot average fitness per population
    plt.plot(average_fitness_per_population_array)
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Average fitness per population')
    plt.savefig(f'{run_number}_average_fitness_per_population.png')
    plt.close()
    # plot best_individual_fitness_per_population
    plt.plot(best_individuals_fitness_per_population_array, 'ro')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Fittest individual per population')
    plt.axis([0, len(best_individuals_fitness_per_population_array), min(best_individuals_fitness_per_population_array) - 5, max(best_individuals_fitness_per_population_array) + 5])
    plt.savefig(f'{run_number}_fittest_individual_per_population.png')
    plt.close()
    # plot standard deviation per population
    plt.plot(standard_deviation_per_population)
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Standard Deviation per population')
    plt.savefig(f'{run_number}_standard_deviation_per_population.png')
    plt.close()
    os.chdir('../../../')
    print(f'check_directory: {os.getcwd()}')

def test_best_individual_five_times(individual, env):
    print('*****Testing five times the best individual ******************')
    best_individual_scores= np.zeros(shape=5)
    for game_runs in range(5):
        individual_value, individual_information = test_individual(individual, env)
        best_individual_scores[game_runs] = individual_information[0]
        print(f'best_individual_score:\n{best_individual_scores}')
    return best_individual_scores

def visualize_box_plot(array, folder_name, algorithm_name ):
    print(f'current_directory to save box plot outside of runs: {os.getcwd()}')
    if os.getcwd() != folder_name:
        os.chdir(os.getcwd()+'/'+algorithm_name+'/'+folder_name)
    print(f'array received: {array}')
    plt.boxplot(array)
    plt.xlabel("Algorithms")
    plt.ylabel('Fitness Value')
    plt.title('Mean Fitness Box Plot of over 5 runs of best individuals')
    plt.savefig(f'{algorithm_name}_box_plot.png')
    plt.close()

def draw_box_plot(average_fitness_all_runs_per_generation, max_fitness_all_runs_per_generation, standard_deviation_average_fitness_all_runs_per_generation, standard_deviation_max_fitness_all_runs_per_generation):
    number_of_generations_array= list(np.arange(len(average_fitness_all_runs_per_generation)))
    plt.errorbar(number_of_generations_array, max_fitness_all_runs_per_generation, yerr= standard_deviation_max_fitness_all_runs_per_generation,ecolor="black", label="Max Fitness")
    plt.errorbar(number_of_generations_array, average_fitness_all_runs_per_generation, yerr= standard_deviation_average_fitness_all_runs_per_generation ,ecolor="black", label="Average Fitness", )
    plt.legend(loc="center")
    plt.title('Average and max fitness of all runs per generation')
    plt.xlabel('Generations')
    plt.savefig('line_plot.png')
    plt.show()

if __name__ == '__main__':
    for enemy in list_of_enemies:
        # env.update_parameter('enemies', [enemy])
        # log state of the env
        # env.state_to_log()
        
        # Line plot array for each generation per run - have run in rows and generations in columns and then pick up columnwise via [:, generation_number]
        max_fitness_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, maximum_generations)) 
        max_individual_value_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, individuals_size_with_sigma))
        mean_fitness_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, maximum_generations))
        standard_deviation_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, maximum_generations))
 
        # ******* Box plot array 
        # create a total_runs array for the five_times run of best_individuals 
        ten_runs_five_times_individuals_arrays = np.zeros(shape=(total_runs))
        for current_run in range(total_runs):
            # keep a counter for the maximum of each generation and change it to this if fitness is more 
            # initialize a best counter score so that the first value is compareable
            max_individual_score_current_run= 0
            print(f'******************************Starting run {current_run} / {total_runs - 1}')
            # keep records of the best individual fitness, best individual, mean and sd of each population
            best_individuals_fitness_populations_array = np.zeros(shape=(maximum_generations))
            best_individuals_value_populations_array = np.zeros(shape=(maximum_generations, individuals_size_with_sigma))
            mean_fitness_populations_array = np.zeros(shape=(maximum_generations))
            standard_deviation_populations_array = np.zeros(shape=(maximum_generations))
            for new_generation in range(maximum_generations):
                print(f'********************Starting generation {new_generation}/{maximum_generations-1} **************************************')
                # add stagnation 
                # if maximum_improvement_counter != improvement_counter:
                if new_generation == 0:
                    # one more value for sigma
                    # generation_population = create_random_uniform_population(population_size, individuals_size)
                    generation_population = create_random_uniform_population(population_size, individuals_size_with_sigma)
                else: 
                    generation_population = create_new_population_two_parents_two_offsprings(combined_list_population_values_and_fitness, generation_population, population_fitness_array, new_generation, maximum_generations)
                # get population information
                
                combined_list_population_values_and_fitness ,population_fitness_array, best_individual_fitness, best_individual_value, average_fitness_population, standard_deviation_population = get_population_information(generation_population, env) 
                
                # box plots 
                best_individuals_fitness_populations_array[new_generation] = best_individual_fitness
                best_individuals_value_populations_array[new_generation] = best_individual_value
                mean_fitness_populations_array[new_generation] = average_fitness_population
                standard_deviation_populations_array[new_generation] = standard_deviation_population
                # line plot arrays
                max_fitness_ten_runs_twenty_generations_array[current_run][new_generation] = best_individual_fitness
                # check about the max individual in the current run
                if best_individual_fitness > max_individual_score_current_run:
                    max_individual_score_current_run = best_individual_fitness
                    # print(f'max_individual_score_current_run: {max_individual_score_current_run}')
                    max_individual_value_ten_runs_twenty_generations_array[current_run] = best_individual_value
                    # print(f'max_individual_value_ten_runs_twenty_generations_array[current_run]: {max_individual_value_ten_runs_twenty_generations_array[current_run]}')
                mean_fitness_ten_runs_twenty_generations_array[current_run][new_generation] = average_fitness_population
                standard_deviation_ten_runs_twenty_generations_array[current_run][new_generation]= standard_deviation_population
                # update solution of environment 
                env.update_solutions([generation_population, population_fitness_array, best_individuals_fitness_populations_array, best_individuals_value_populations_array, mean_fitness_populations_array, standard_deviation_populations_array]) 

            print("*******************************************************STATISTICS*****************************")
            print(f'\nbest_individuals_fitness_populations_array:\n{best_individuals_fitness_populations_array}')
            print(f'\nbest_individuals_value_populations_array:\n{best_individuals_value_populations_array}')
            print(f'\nmean_fitness_populations_array:\n{mean_fitness_populations_array}')
            print(f'\nstandard_deviation_populations_array:\n{standard_deviation_populations_array}')
            # Box Plot Data
                # find best individual of all generations
            best_individual_fitness_position_all_generations = np.argmax(best_individuals_fitness_populations_array)
            print(f'\nbest_individual_fitness_position_all_generations:\n{best_individual_fitness_position_all_generations}')
            best_individual_value_all_generations= best_individuals_value_populations_array[best_individual_fitness_position_all_generations]
            print(f'\nbest_individual_value_all_generations:\n{best_individual_value_all_generations}')

            # Since we have the big arrays for all runs per generation, consider taking the values from the overall instead of having twice the arrays
            # try to get fittest individual from all generations of each run 
            print(f'max_fitness_ten_runs_twenty_generations_array:\n {max_fitness_ten_runs_twenty_generations_array}')
            
            # go over each run and find the best individual of all generations and add it to max_fitness
            max_fitness_all_generations_per_run= np.zeros(shape=total_runs)
            for run_counter in range(total_runs):
                # current run of max fitness of all generations
                current_run_max_array=max_fitness_ten_runs_twenty_generations_array[run_counter]
                print(f'current_run_array: {current_run_max_array}')
                current_run_max_value_position= np.argmax(current_run_max_array)
                max_value_current_run= current_run_max_array[current_run_max_value_position]
                print(f'max_value_current_run: {max_value_current_run}')
                max_fitness_all_generations_per_run[run_counter]= max_value_current_run

                # current run of mean fitness of all generations
                # current_run_average_fitness_array= mean_fitness_ten_runs_twenty_generations_array[run_counter]
                # print(f'current_run_average_fitness_array: {current_run_average_fitness_array}')
                # current_run_average_fitness= np.average(current_run_max_array)
                # average_fitness_all_generations_per_run[run_counter]= current_run_average_fitness
            
            # only the max fitness of all generations per run
            print(f'max_fitness_all_generations_per_run:\n{max_fitness_all_generations_per_run}')
            # play 5 times with the max fitness individual - ****need to get the max individual 

            # mean fitness of all 
            
            max_fitness_first_generation_all_runs= max_fitness_ten_runs_twenty_generations_array[:, 0]
            print(f'max_fitness_first_generation_all_runs: {max_fitness_first_generation_all_runs}')

            print(f'mean_fitness_ten_runs_twenty_generations_array: {mean_fitness_ten_runs_twenty_generations_array}')

            #******* Line plot Average and max  # COnsider make box plot with line plot arrays

            # get average fitness and standarddeviation for all runs per generation 
            average_fitness_all_runs_per_generation= np.zeros(shape=maximum_generations)
            standard_deviation_average_fitness_all_runs_per_generation= np.zeros(shape=maximum_generations)
            

            # get max fitness and standard deviation for all runs per generation
            max_fitness_all_runs_per_generation= np.zeros(shape= maximum_generations)
            standard_deviation_max_fitness_all_runs_per_generation= np.zeros(shape=maximum_generations)

            for generation_counter in range(maximum_generations):
                # average fitness 
                current_generation_average_fitness= mean_fitness_ten_runs_twenty_generations_array[:, generation_counter]
                # print(f'current_generation_average_fitness: {current_generation_average_fitness}')
                average_fitness_all_runs_per_generation[generation_counter]= np.mean(current_generation_average_fitness)
                standard_deviation_average_fitness_all_runs_per_generation[generation_counter]= np.std(current_generation_average_fitness)

                # max fitness 
                current_generation_max_fitness= max_fitness_ten_runs_twenty_generations_array[:, generation_counter]
                # print(f'current_generation_max_fitness: {current_generation_max_fitness}')
                max_fitness_all_runs_per_generation[generation_counter]= np.max(current_generation_max_fitness)
                standard_deviation_max_fitness_all_runs_per_generation[generation_counter]= np.std(current_generation_max_fitness)

            print(f'average_fitness_all_runs_per_generation: {average_fitness_all_runs_per_generation}')
            print(f'standard_deviation_average_fitness_all_runs_per_generation: {standard_deviation_average_fitness_all_runs_per_generation}')
            print(f'max_fitness_all_runs_per_generation: {max_fitness_all_runs_per_generation}')
            print(f'standard_deviation_max_fitness_all_runs_per_generation: {standard_deviation_max_fitness_all_runs_per_generation}')

            # Draw Box Plot and save in current folder 
            draw_box_plot(average_fitness_all_runs_per_generation, max_fitness_all_generations_per_run, standard_deviation_average_fitness_all_runs_per_generation, standard_deviation_max_fitness_all_runs_per_generation)

                # test that individual 5 times and get the mean of those 5 
            five_times_scores_best_individual_array= test_best_individual_five_times(best_individual_value_all_generations, env)
                # get the mean of the five_times_scores
            mean_of_five_times_scores_best_individual = np.mean(five_times_scores_best_individual_array)
            print(f'mean_of_five_times_scores_best_individual: {mean_of_five_times_scores_best_individual}')

                # add that value in the array for all runs regarding the mean_five_times_score_best_individual
            ten_runs_five_times_individuals_arrays[current_run] = mean_of_five_times_scores_best_individual
            #save the arrays 
            working_directory = save_array_to_files_with_defined_parameters(experiment_name, working_directory, total_runs, current_run, maximum_generations, population_size,  best_individuals_fitness_populations_array, best_individuals_value_populations_array, mean_fitness_populations_array, standard_deviation_populations_array)

            print("*******************************************Creating Graphs**************************************")
            create_graphs_for_each_run(mean_fitness_populations_array, best_individuals_fitness_populations_array, standard_deviation_populations_array, experiment_name, working_directory, current_run)

        # get the array of 10 means of the 10 runs of the best_individual
        print(f'ten_runs_five_times_individuals_arrays\n{ten_runs_five_times_individuals_arrays}')
        # compute a box plot from the array of 10 runs for the algorithm
        visualize_box_plot(ten_runs_five_times_individuals_arrays, working_directory, experiment_name)
