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

# REMEMBER TO REMOVE
# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

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

# size of individuals
individuals_size = (env.get_num_sensors()+1)*hidden_neurons + (hidden_neurons+1)*5
# population size made of n individuals
population_size = 20
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
    combined_list_population_individual_fitness= []
    population_fitness_array = np.zeros(shape=population.shape[0])
    for individual_position in range(population.shape[0]):
        # check if i ever need individual 
        current_individual, individual_information= test_individual(population[individual_position], env)
        # print(f'individual_information: {individual_information} [0]: {individual_information[0]}')
        population_fitness_array[individual_position] = individual_information[0]
        combined_list_population_individual_fitness.append([current_individual, individual_information[0]]) 
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
    # print(f'combined_list_population_individual_fitness: {combined_list_population_individual_fitness}')
    
    return combined_list_population_individual_fitness, population_fitness_array, most_fit_value, most_fit_individual, mean_fitness_population, standard_deviation_population

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
    # number_of_offspring_pairs= 1
    # print(f'parent_one: {first_parent}\nand shape: {first_parent.shape[0]}')
    # two children
    # number_of_offsprings_array = np.zeros(shape=(number_of_offspring_pairs * 2, first_parent.shape[0]))
    # print(f'number_of_offsprings_array\n{number_of_offsprings_array}\nand shape: {number_of_offsprings_array.shape}')
    # for pair_position in range (number_of_offspring_pairs):
    alpha_parameter= np.random.uniform(probability_lower_bound, probability_upper_bound)        
        # print(f'alpha_parameter: {alpha_parameter}')
    first_offspring= first_parent * alpha_parameter + second_parent * (1-alpha_parameter)
    second_offspring= second_parent * alpha_parameter + first_parent * (1-alpha_parameter)
    # print(f'number_of_offsprings_array: {number_of_offsprings_array}')
    return first_offspring, second_offspring


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
    print(f'normalized_array_divided_by_sum:\n{normalized_array_divided_by_sum}')   
    print(f'shape: {normalized_array_divided_by_sum.shape}')

    # randomly choose mating pool of parents given the array of parents, the weights and selecting k number with replacement 
    stochastic_choice_of_parents= choices(received_list, weights=normalized_array_divided_by_sum, k=number_of_individuals_to_select)
    print(f'stochastic_choice_of_parents: {stochastic_choice_of_parents}')
      

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
def cumulative_distribution_function_prioritizing_worst(received_list, number_of_individuals_to_select ):
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
    print(f'sigma_scaling_array:\n {sigma_scaling_array}')

    # Min Max Normalization from scikit 
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    min_max_scaler= MinMaxScaler()
    # reshaped array row-wise to feed to min_max_scalar
    # normalized values
    # print(f'sigma_scaling_array.reshape(-1, 1):\n {sigma_scaling_array.reshape(-1, 1)}')
    normalized_array= min_max_scaler.fit_transform(sigma_scaling_array.reshape(-1, 1))
    print(f'normalized_array:\n {normalized_array}')

    normalized_array= np.where(normalized_array==0, min_value,normalized_array)
    print(f'after replacement normalized_array: {normalized_array}')
    # sum of array for denominator
    sum_normalized= np.sum(normalized_array)
    # print(f'sum_normalized: {sum_normalized}')
    # divide each probability by the total to get cdf 
    normalized_array_divided_by_sum= normalized_array / sum_normalized
    # print(f'normalized_array_divided_by_sum:\n{normalized_array_divided_by_sum}')   
    # print(f'shape: {normalized_array_divided_by_sum.shape}')

    # randomly choose mating pool of parents given the array of parents, the weights and selecting k number with 
    index_to_select_list= random.choices(range(len(received_list)), weights=normalized_array_divided_by_sum, k=number_of_individuals_to_select )
    print(f'index_to_select_list: {index_to_select_list}')

    return index_to_select_list



    


def non_uniform_mutation_varying_sigma_mutate_individual(individual, probability_of_mutation, current_generation, total_number_of_generations):
    # non-uniform mutation 
    # the idea is to start with a high sigma value and as generations proceed to reduce the step --> sigma = 1 - (current/ max)
    # current generations will be 1 instead of 0 as 0 is creation of uniform population thus -1 for the sake of starting with 1 as sigma
    varying_sigma_value = 1 - ((current_generation - 1) / total_number_of_generations)
    # print(f'varying_sigma: {varying_sigma_value}')
    # create the random probabilities for the individual from before
    random_uniform_mutation_probability_array= np.random.uniform(probability_lower_bound, probability_upper_bound, individual.shape[0])
    # print(f'random_uniform_mutation_probability_array: {random_uniform_mutation_probability_array}')
    # print(f'random_probability: {random_probability}')
    # if prob > 0.25 
    for individual_position, individual_value in enumerate(individual):
        # if probability is less than the mutation probability then mutate
        if random_uniform_mutation_probability_array[individual_position] < probability_of_mutation:
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
    
def create_new_population_two_parents_two_offsprings(list_population_values_fitness, old_population, cur_generation, max_generations):
    # cross over alpha parameter varying to create more offsprings. sharing most / a lot of information in each child 
    # cross_over_alpha_parameter_array= np.array([0.1, 0.25, 0.4])
    # print(f'old_population.shape: {old_population.shape} and values: {old_population}')

    # same size population is maintained 
    next_generation_population=np.zeros_like(old_population)
    # print(f'next_generation_population.shape: {next_generation_population.shape} and values: {next_generation_population}')

    # stochastically create the mating population of k individuals by using sigma scaling and normalization
    mating_population= choose_k_individuals_for_mating_stochastically_sigma_scaling(list_population_values_fitness,4)
    print(f'mating_population: {mating_population}')
    # offspring array from pairings of mating population
    offspring_population_array= np.zeros(shape=(len(mating_population), len(mating_population[0][0])))
    print(f'offspring_population_array.shape: {offspring_population_array.shape}')
        # go over the list pair_wise and select the parents 
    for position_counter in range(0, len(mating_population) - 1, 2):
        # get value of individual parent and disregard fitness
        parent_one= mating_population[position_counter][0]
        print(f'parent_one: {parent_one}')
        parent_two= mating_population[position_counter + 1][0]
        print(f'parent_two: {parent_two}')
        offspring_one, offpsring_two= crossover_two_parents_alpha_uniform(parent_one, parent_two)
        mutated_offspring_one= non_uniform_mutation_varying_sigma_mutate_individual(offspring_one, mutation_probability, cur_generation, max_generations)
        print(f'mutated_offspring_one:\n{mutated_offspring_one}\nand shape: {mutated_offspring_one.shape}')
        mutated_offspring_two= non_uniform_mutation_varying_sigma_mutate_individual(offpsring_two, mutation_probability, cur_generation, max_generations)
        print(f'mutated_offspring_one:\n{mutated_offspring_two}\nand shape: {mutated_offspring_two.shape}')
        offspring_population_array[position_counter]= mutated_offspring_one
        offspring_population_array[position_counter + 1]= mutated_offspring_two

    print(f'offspring_population_array:\n {offspring_population_array}')
    # Assumption that the children will be fitter than the worst individual
    # depending on the generations you want to provide more selection pressure - initially just replace the worst only by one of the offsprings and then replace 
    # all the worst ones. Half way through start replacing more than the worst
    # find worst individual 
    

    if cur_generation < max_generations/2 :
        # number of individuals to replace
        number_of_individuals_to_replace= 1
    else:
        number_of_individuals_to_replace= len(offspring_population_array)
    positions_of_individual_to_replace= cumulative_distribution_function_prioritizing_worst(list_population_values_fitness, number_of_individuals_to_replace)
    for position in range(len(positions_of_individual_to_replace)):
        print(f'positions_of_individual_to_replace[position]: {positions_of_individual_to_replace[position]}')
        element_to_remove= list_population_values_fitness[positions_of_individual_to_replace[position]]
        print(f'element_to_remove: {element_to_remove}')
        # replace the individuals with sampling from the offsprings
        # anti lista me array. *************************************
        list_population_values_fitness[positions_of_individual_to_replace[position]] = sample(offspring_population_array) 

        print(f'received_list[index_to_select[position]]: {received_list[index_to_select[position]]}')
        print('replace only the worst')

        index_of_individual = list_population_values_fitness.index(individual_to_replace)
        print(f'index_of_individual: {index_of_individual}')
        print(f'list_population_values_fitness[0]: {list_population_values_fitness[0]}')
        get_only_individuals= list_population_values_fitness[0]
        print(f'get_only_individuals:\n{get_only_individuals}')

        # randomly one of the offsprings to replace the worst 
        random_choice_of_offspring= sample(offspring_population_array) 
        print(f'random_choice_of_offspring: {random_choice_of_offspring}')
        print(f'old_population:\n{old_population}')
        

      


    # given

        ## continue from here survivor with stochastically again 
        

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

    return next_generation_population


# mutation --> random < 0.5 then modify sigma

# write numpy arrays to files 
def save_array_to_files_with_defined_parameters(experiment_name_folder, folder_name, number_of_runs, current_run_value, number_of_generations, number_of_individuals, best_individuals_fitness_per_population, best_individual_per_population_array, average_fitness_per_population, standard_deviation_per_population):
    # get time to attach it to the folder 
    from datetime import datetime
    if folder_name == "not_yet":
        folder_name = (f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S_')}{number_of_runs}number_of_runs_{number_of_generations}_generations_{number_of_individuals}_number_of_individuals")
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

def draw_line_plot(average_fitness_all_runs_per_generation, max_fitness_all_runs_per_generation, standard_deviation_average_fitness_all_runs_per_generation, standard_deviation_max_fitness_all_runs_per_generation):
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
        max_individual_value_ten_runs_twenty_generations_array= np.zeros(shape=(total_runs, individuals_size))
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
            best_individuals_value_populations_array = np.zeros(shape=(maximum_generations, individuals_size))
            mean_fitness_populations_array = np.zeros(shape=(maximum_generations))
            standard_deviation_populations_array = np.zeros(shape=(maximum_generations))
            for new_generation in range(maximum_generations):
                print(f'********************Starting generation {new_generation}/{maximum_generations-1} **************************************')
                # add stagnation 
                # if maximum_improvement_counter != improvement_counter:
                if new_generation == 0:
                    generation_population = create_random_uniform_population(population_size, individuals_size)
                else: 
                    generation_population = create_new_population_two_parents_two_offsprings(combined_list_population_values_and_fitness, generation_population, new_generation, maximum_generations)
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

            # Per run arrays - save the arrays 
            working_directory = save_array_to_files_with_defined_parameters(experiment_name, working_directory, total_runs, current_run, maximum_generations, population_size,  best_individuals_fitness_populations_array, best_individuals_value_populations_array, mean_fitness_populations_array, standard_deviation_populations_array)

            print("*******************************************Creating Graphs**************************************")
            create_graphs_for_each_run(mean_fitness_populations_array, best_individuals_fitness_populations_array, standard_deviation_populations_array, experiment_name, working_directory, current_run)

            # Box Plot Data
                # find best individual of all generations
            best_individual_fitness_position_all_generations = np.argmax(best_individuals_fitness_populations_array)
            # print(f'\nbest_individual_fitness_position_all_generations:\n{best_individual_fitness_position_all_generations}')
            best_individual_value_all_generations= best_individuals_value_populations_array[best_individual_fitness_position_all_generations]
            # print(f'\nbest_individual_value_all_generations:\n{best_individual_value_all_generations}')

            # Since we have the big arrays for all runs per generation, consider taking the values from the overall instead of having twice the arrays
            # try to get fittest individual from all generations of each run 
            print(f'max_fitness_ten_runs_twenty_generations_array:\n {max_fitness_ten_runs_twenty_generations_array}')
            
        # Once all runs have finished get the information 
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

        # Draw Line Plot and save in current folder 
        draw_line_plot(average_fitness_all_runs_per_generation, max_fitness_all_runs_per_generation, standard_deviation_average_fitness_all_runs_per_generation, standard_deviation_max_fitness_all_runs_per_generation)

            # test that individual 5 times and get the mean of those 5 
        five_times_scores_best_individual_array= test_best_individual_five_times(best_individual_value_all_generations, env)
            # get the mean of the five_times_scores
        mean_of_five_times_scores_best_individual = np.mean(five_times_scores_best_individual_array)
        print(f'mean_of_five_times_scores_best_individual: {mean_of_five_times_scores_best_individual}')

            # add that value in the array for all runs regarding the mean_five_times_score_best_individual
        ten_runs_five_times_individuals_arrays[current_run] = mean_of_five_times_scores_best_individual

        # get the array of 10 means of the 10 runs of the best_individual
        print(f'ten_runs_five_times_individuals_arrays\n{ten_runs_five_times_individuals_arrays}')
        # compute a box plot from the array of 10 runs for the algorithm
        visualize_box_plot(ten_runs_five_times_individuals_arrays, working_directory, experiment_name)
