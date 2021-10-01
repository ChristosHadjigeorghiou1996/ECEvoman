import os 
def load_numpy_files():
    print(f'current_directory: {os.getcwd()}')

    # print(f'new_directory: {os.getcwd()}')
    # os.chdir(path_till_numpy_files)
    average_fitness_per_population_array_name = "average_fitness_per_population.npy"
    average_fitness_per_population_array = np.load(average_fitness_per_population_array_name)
    # print(f'average_fitness_per_population_array:\n{average_fitness_per_population_array}')
    best_individuals_fitness_per_population_array_name = "best_individuals_fitness_per_population.npy"
    best_individuals_fitness_per_population_array = np.load(best_individuals_fitness_per_population_array_name)
    # print(f'best_individuals_fitness_per_population_array:\n{best_individuals_fitness_per_population_array}')
    # array with the best individual per population
    best_individual_per_population_array_name = "best_individual_per_population_array.npy"
    best_individual_per_population_array = np.load(best_individual_per_population_array_name)
    # print(f'best_individual_per_population_array:\n{best_individual_per_population_array}')
    # array of standard_deviations of all populations 
    standard_deviation_per_population_array_name = "standard_deviation_per_population.npy"
    standard_deviation_per_population_array = np.load(standard_deviation_per_population_array_name)
    # print(f'standard_deviation_per_population_array:\n{standard_deviation_per_population_array}')
    return average_fitness_per_population_array, best_individuals_fitness_per_population_array, best_individual_per_population_array, standard_deviation_per_population_array   
