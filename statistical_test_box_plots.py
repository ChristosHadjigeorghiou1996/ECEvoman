import numpy as np 
import os
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# load the two box plot arrays 

def load_numpy_files(file_name):
    # print(f'Current directory: {os.getcwd()}')
    # print(f'filename:{file_name}')
    loaded_array = np.load(file_name)
    # print(f'loaded_array:\n{loaded_array}')
    return loaded_array


def visualize_all_six_plots(array_alg_a, array_alg_b, enemy_list):
    box_plot_dict= {}
    for counter in range(len(enemy_list)): 
        box_plot_dict[f"alg_a_{enemy_list[counter]}"] = array_alg_a[counter]
        box_plot_dict[f"alg_b_{enemy_list[counter]}"] = array_alg_b[counter]
        if counter == len(enemy_list) - 1:
            # print(f'box_plot_dict:\n{box_plot_dict}')
            fig, ax1 = plt.subplots()
            ax1.boxplot(box_plot_dict.values())
            ax1.set_xticklabels(box_plot_dict.keys())
            fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
            ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
            ax1.set(
            axisbelow=True,  # Hide the grid behind plot objects
            title='Mean Fitness Over 5 Runs of Best Individuals Per Enemy',
            xlabel='Enemy Fought Against',
            ylabel='Fitness',
            )
            plt.savefig(f'both_algorithms_all_enemies_box_plot.png')
            plt.close()

def get_t_test_values(array_alg_a, array_alg_b, enemy_list):
    # wilcoxon test considers the difference between values
    # create numpy array for faster computation
    differences_per_enemy= array_alg_a - array_alg_b
    print(f'Difference of plot values per enemy from {enemy_list}:\n{differences_per_enemy}')
    print(f"Testing with wilcoxon test")
    for enemy_counter in range(len(enemy_list)):
        # print(f'enemy: {enemy_list[enemy_counter]} : {differences_per_enemy[enemy_counter]}')
        w, p = wilcoxon(differences_per_enemy[enemy_counter])
        print(f'enemy: {enemy_list[enemy_counter]} -> w: {w}, p: {p}')
        print(f"Null hypothesis as no difference in the mean between two algorithms and H1 that the are different")
        if p > 0.05:
            print(f"Cannot reject null hypothesis that there is a difference according to mean given a 5% CI")
        else:
            print(f"Reject null hypothesis given a 5% CI as there is a difference between algorithm_a and algorithm_b for enemy: {enemy_list[enemy_counter]}")
        # test median of differences, wilcoxon greater -? null hypothesis is negative, if <5% CI -> in favour of alternative thus positive median 
        w_median, p_median= wilcoxon(differences_per_enemy[enemy_counter], alternative="greater")
        print(f'enemy: {enemy_list[enemy_counter]} -> w_median: {w_median}, p_median: {p_median}')
        print(f"Test median of differences, Null hypothesis is negative median while alternative hypothesis indicates positive median, 5% CI ")
        if p_median > 0.05:
            print(f"Cannot reject null hypothesis, negative median given a 5% CI")
        else:
            print(f"Reject null hypothesis, positive median is indicated given a 5% CI that median is greater than 0 between algorithm_a and algorithm_b for enemy: {enemy_list[enemy_counter]}")


if __name__ == "__main__":
    print('searching for arrays')
    # enemy_list= [2, 5, 3]
    enemy_list= [8]
    alg_a_plot_array= load_numpy_files("alg_a_enemy_8_box_plot_array.npy")
    alg_b_plot_array= load_numpy_files("alg_b_enemy_8_box_plot_array.npy")
    visualize_all_six_plots(alg_a_plot_array, alg_b_plot_array, enemy_list)      
    get_t_test_values(alg_a_plot_array, alg_b_plot_array, enemy_list)



