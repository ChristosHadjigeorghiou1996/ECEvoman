################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from numpy.core.fromnumeric import shape
from numpy.lib import load
sys.path.insert(0, 'evoman') 
from demo_controller import player_controller
from environment import Environment
# import numpy for random initialization and arrays
import numpy as np 


# general infromation 
experiment_type = "test"
# experiment_type = "test_b"
experiment_name = "algorithm_a" 
experiment_mode = "single"
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
population_size = 10
# max generations to run
maximum_generations = 5
# max iterations to run without improvement to indicate stagnation
improvement_value = 0
improvement_counter = 0 # cap it to 15
maximum_improvement_counter = 15; # if counter reach break due to stagnation
# parameters for random initialization being between -1 and 1 to be able to change direction
lower_limit_uniform_population = -1
upper_limit_uniform_population = 1
# trim lower bound to 0 and upper bound to 1 for mutation
lower_bound = 0
upper_bound = 1
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
    new_population = np.random.uniform(lower_limit_uniform_population, upper_limit_uniform_population, (size_of_populations, size_of_individuals))
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
    population_fitness_array = np.zeros(shape=population.shape[0])
    for individual_position in range(population.shape[0]):
        # check if i ever need individual 
        current_individual, individual_information= test_individual(population[individual_position], env)
        # print(f'individual_information: {individual_information} [0]: {individual_information[0]}')
        population_fitness_array[individual_position] = individual_information[0]
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
    
    return population_fitness_array, most_fit_value, most_fit_individual, mean_fitness_population, standard_deviation_population



def find_best_individual(population, env):
    # mean fitness of population -> sum the individuals and divide by the length of the population
    sum_fitness_population = 0  
    # section for best individual per populations
    best_individual_fitness = -10
    best_individual=population[0]
    best_individual, best_individual_information= test_individual(best_individual, env) 

    for individual in population:
        current_individual, individual_information= test_individual(individual, env)
        # get the fitness of the individual and add it to the sum
        sum_fitness_population = sum_fitness_population + individual_information[0]
        # print(f'individual_information[0]: {individual_information[0]} and type: {type(individual_information[0])}'    )
        if best_individual_fitness < individual_information[0]:
            # print(f'new_best: {individual_information}')
            best_individual_fitness= individual_information[0]
            best_individual = current_individual
            best_individual_information = individual_information
    # print(f'best_individual: {best_individual}')
    # print(f'best_individual_information: {best_individual_information}')

    # divide the sum of the fitness by the number of individuals
    mean_fitness_population = sum_fitness_population / population.shape[0]
    return best_individual, best_individual_information, mean_fitness_population


# randomly select two parents in the population and combine to produce two children with param 0.4 
# instead of producing two children, produce more lets say 4-6 children and then populate to decide on the fitter version 
def crossover_two_parents_two_children(first_parent, second_parent, alpha_parameter, env):
    # indicated by paper for balancing exploration and exploitation
    # alpha_parameter = 0.4
    # arithmetic average
    # print(f'P1: {first_parent}, P2: {second_parent}')
    first_offspring = first_parent*alpha_parameter + (1-alpha_parameter) * second_parent 
    # print(f'first_offspring: {first_offspring}')
    # instead of testing now, just give the two offsprings and test once at the end the entire population
    # first_offspring_values, first_offspring_information = test_individual(first_offspring, env)
    second_offspring = second_parent * alpha_parameter + ( 1 - alpha_parameter) * first_parent
    # print(f'second_offspring: {second_offspring}')
    # second_offspring_values, second_offspring_information = test_individual(second_offspring, env)
    # print(f'second_offspring_information: {second_offspring_information}')
    # return first_offspring_values, first_offspring_information, second_offspring_values, second_offspring_information
    return first_offspring, second_offspring

# def blend_crossover
#     # abs distance between parents 
    

# randomly select two parents and ensure they are not the same from given population
def randomly_select_two_individuals(provided_population):
    # print(f'provided_population.shape: {provided_population.shape}')
    parent_one_position = np.random.randint(0, provided_population.shape[0])
    parent_one= provided_population[parent_one_position]
    # print(f'parent_one_position: {parent_one_position} which is: \n {parent_one}')
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

def mutate_individual(individual):
    random_probability = np.random.random() 
    # print(f'random_probability: {random_probability}')
    # if prob > 0.25 
    if random_probability > mutation_probability:
        # print('mutate individual and play game') 
        # mutate by considering some noise drawn from Gaussian distribution and then check if the random probability is > 0.5 in which case add otherwise subtract 
        if random_probability > 0.5: 
            mutated_individual = individual + np.random.normal(0, 1, individual.shape)
        else:
            mutated_individual = individual - np.random.normal(0, 1, individual.shape)
        # print(f'mutated individual: {mutated_individual}')
        return mutated_individual
    return individual
        # maybe consider uniform from -1 to 1 instead 
    
def create_new_population(old_population, environment):
    # print(f'old_population.shape: {old_population.shape} and values: {old_population}')
    next_generation_population=np.zeros_like(old_population)
    # print(f'next_generation_population.shape: {next_generation_population.shape} and values: {next_generation_population}')
    # print(f'type of old_population: {type(old_population)} and next_gen: {type(next_generation_population)}')
    # go two steps each time removing two parents and creating two offsprings which are added to the new population. Stop until we are done whichi s of same size.
    for individual_position in range(0, old_population.shape[0], 2):
        # select two parents from the population ensuring they are not the same and forgetting them from the population which assumes they reproduce once 
        parent_one, parent_two, old_population= randomly_select_two_individuals(old_population)
        # crossover with arithmetic average with alpha 0.4 
        # offspring_one, offspring_one_information, offspring_two, offspring_two_information = crossover_two_parents_two_children(parent_one, parent_two, crossover_alpha_parameter, environment)
        offspring_one, offspring_two= crossover_two_parents_two_children(parent_one, parent_two, crossover_alpha_parameter, environment)
        # undergo process of mutation
        modified_offspring_one =mutate_individual(offspring_one)
        # print(f'modified_offspring_one type: {type(modified_offspring_one)} and shape: {modified_offspring_one.shape}')
        modified_offspring_two = mutate_individual(offspring_two)
        next_generation_population[individual_position]= modified_offspring_one
        next_generation_population[individual_position + 1]= modified_offspring_two
    
    # print(f'next_generation_population: {next_generation_population} and shape: {next_generation_population.shape}')

    return next_generation_population


# mutation --> random < 0.5 then modify sigma

# write numpy arrays to files 
def save_array_to_files_with_defined_parameters(experiment_name_folder, number_of_generations, number_of_individuals, best_individuals_fitness_per_population, best_individual_per_population_array, average_fitness_per_population, standard_deviation_per_population):
    # get time to attach it to the folder 
    from datetime import datetime
    folder_name = (f"{datetime.now().strftime('%H_%M_%S_')}{number_of_generations}_generations_{number_of_individuals}number_of_individuals")
    # # create and navigate to the experiment_name folder 
    if not os.path.exists(experiment_name_folder):
        os.makedirs(experiment_name_folder)
    os.chdir(os.getcwd() +'/'+experiment_name_folder)
    print(f'working directory: {os.getcwd()}')
    # create folder with time and specific tuple 
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        os.chdir(os.getcwd() + '/'+ folder_name)
    print(f'current directory to save the arrays: {os.getcwd()}')
    # save the numpy arrays individually 
    np.save('best_individuals_fitness_per_population', best_individuals_fitness_per_population)
    np.save('best_individual_per_population_array', best_individual_per_population_array)
    np.save('average_fitness_per_population', average_fitness_per_population)
    np.save('standard_deviation_per_population', standard_deviation_per_population)

def load_numpy_files(directory):
    print(os.getcwd())
    os.chdir(os.getcwd() +'/algorithm_a/18_38_33__5_generations_10number_of_individuals')
    # directory="C:\Users\hadji\Documents\Amsterdam\Year 2\Evolutionary Computing\Assignment\ECEvoman\algorithm_a\18_38_33__5_generations_10number_of_individuals"

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

def visualize_average_fitness_per_population_graph(array):
    # import pyplot to visualize arrays and create graphs
    import matplotlib.pyplot as plt
    print(f'array received: {array}')
    plt.plot(array)
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Average fitness per population')
    plt.savefig('average_fitness_per_population.png')
    plt.show()

def visualize_best_individuals_fitness_per_population_array(array):
    # import pyplot to visualize arrays and create graphs
    import matplotlib.pyplot as plt
    print(f'array received: {array}')
    plt.plot(array, 'ro')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Fittest individual per population')
    plt.axis([0, len(array), min(array) - 5, max(array) + 5])
    plt.savefig('fittest_individual_per_population.png')
    plt.show()

    
def visualize_standard_deviation_per_population_graph(array):
    # import pyplot to visualize arrays and create graphs
    import matplotlib.pyplot as plt
    print(f'array received: {array}')
    plt.plot(array)
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Standard Deviation per population')
    plt.savefig('standard_deviation_per_population.png')
    plt.show()


if __name__ == '__main__':
    for enemy in list_of_enemies:
        # env.update_parameter('enemies', [enemy])
        # log state of the env
        # env.state_to_log()

        # # keep records of the best individual fitness, best individual, mean and sd of each population
        # best_individuals_fitness_populations_array = np.zeros(shape=(maximum_generations))
        # best_individuals_value_populations_array = np.zeros(shape=(maximum_generations, individuals_size))
        # mean_fitness_populations_array = np.zeros(shape=(maximum_generations))
        # standard_deviation_populations_array = np.zeros(shape=(maximum_generations))
        # for new_generation in range(maximum_generations):
        #     print(f'********************Starting generation {new_generation}/{maximum_generations-1} **************************************')
        #     # add stagnation 
        #     # if maximum_improvement_counter != improvement_counter:
        #     if new_generation == 0:


        #         # print(f'best_individuals_array:\n{best_individuals_array}\n and shape: {best_individuals_array.shape}')
        #         generation_population = create_random_uniform_population(population_size, individuals_size)
        #         # best_individual, best_individual_information, mean_fitness_population = find_best_individual(generation_population, env)
        #         # save the values to the respective arrays
        #         # best_individuals_array[new_generation] = best_individual_information
        #         # print(f'best_individuals_array:\n{best_individuals_array}')
        #         # mean_fitness_populations_array[new_generation] = mean_fitness_population
        #         # print(f'mean_fitness_populations_array:\n{mean_fitness_populations_array}')
        #         population_fitness_array, best_individual_fitness, best_individual_value, average_fitness_population, standard_deviation_population = get_population_information(generation_population, env) 
        #         best_individuals_fitness_populations_array[new_generation] = best_individual_fitness
        #         best_individuals_value_populations_array[new_generation] = best_individual_value
        #         mean_fitness_populations_array[new_generation] = average_fitness_population
        #         standard_deviation_populations_array[new_generation] = standard_deviation_population
                
        #         continue
        #     # print(f'before current generation best_individuals_array:\n{best_individuals_array}')
        #     # print(f'old_population:\n{generation_population}')
        #     generation_population = create_new_population(generation_population, env)
        #     # print(f'new_population:\n{generation_population}')
        #     # print(f'generation_population.shape:{generation_population.shape}')
            
        #     # add the values to the respective arrays
        #     # best_individual, best_individual_information, mean_fitness_population = find_best_individual(generation_population, env)
        #     # best_individuals_array[new_generation] = best_individual_information
        #     # mean_fitness_populations_array[new_generation] = mean_fitness_population
        #     population_fitness_array, best_individual_fitness, best_individual_value, average_fitness_population, standard_deviation_population = get_population_information(generation_population, env) 
        #     best_individuals_fitness_populations_array[new_generation] = best_individual_fitness
        #     best_individuals_value_populations_array[new_generation] = best_individual_value
        #     mean_fitness_populations_array[new_generation] = average_fitness_population
        #     standard_deviation_populations_array[new_generation] = standard_deviation_population
        #     # print(f'best_individuals_array:\n{best_individuals_array}')
        
        # print("*******************************************************STATISTICS*****************************")
        # print(f'\nbest_individuals_fitness_populations_array:\n{best_individuals_fitness_populations_array}')
        # print(f'\nbest_individuals_value_populations_array:\n{best_individuals_value_populations_array}')
        # print(f'\nmean_fitness_populations_array:\n{mean_fitness_populations_array}')
        # print(f'\nstandard_deviation_populations_array:\n{standard_deviation_populations_array}')
        # #save the arrays 
        # save_array_to_files_with_defined_parameters(experiment_name, maximum_generations, population_size,  best_individuals_fitness_populations_array, best_individuals_value_populations_array, mean_fitness_populations_array, standard_deviation_populations_array)

        # instead of running the algorithm try to visualize it and create charts from the numpy arrays
        average_fitness_per_population_array, best_individuals_fitness_per_population_array, best_individual_per_population_array, standard_deviation_per_population_array = load_numpy_files('directory')
        visualize_average_fitness_per_population_graph(average_fitness_per_population_array)
        visualize_best_individuals_fitness_per_population_array(best_individuals_fitness_per_population_array)
        visualize_standard_deviation_per_population_graph(standard_deviation_per_population_array)


                # break
                # # if improvement value stays the same, increment the counter as there is no improvement. Ensure that there is improvement with mutation & crossover
                # if improvement_value == best_individual[1]:
                #     improvement_counter = improvement_counter + 1
                #     if improvement_counter == maximum_improvement_counter:
                #         break
                # else:
                #     improvement_value= best_individual[1]
                #     improvement_counter = 0

                # break

