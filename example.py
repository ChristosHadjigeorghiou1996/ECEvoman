import numpy as np
from numpy.random.mtrand import uniform

min_value = 0.0001
sorted_list = [-1, -5, 3, 6, 10, 0]
prob_list= list(np.arange(len(sorted_list)))
print(f'sorted_list: {sorted_list}')
for position, value in enumerate(sorted_list):
    # print(f'position: {position}')
    # print(f'value: {value}')
    print(f'position:{position} and value: {value}')
    if value <= 0: 
        sorted_list[position]= min_value
print(f'sorted_list: {sorted_list}')
x_min= min(sorted_list)
print(f'x_min: {x_min}')
x_max = max(sorted_list)
print(f'x_max: {x_max}')
# x_new = (x_i - x_min) / x_max - x_min 
for position, value in enumerate(sorted_list):
    new_value= (value - x_min) / (x_max - x_min)
    print(f'new_value: {new_value}')
    if new_value == 0:
        new_value = min_value
    prob_list[position] = new_value
print(f'prob_list: {prob_list}')
sum_of_prob_list = sum(prob_list)
print(f'sum_of_prob_list: {sum_of_prob_list}')
probability_to_select_parent= [(element / sum_of_prob_list) for element in prob_list]
print(f'probability_to_select_parent: {probability_to_select_parent}')
print(sum(probability_to_select_parent))

uniform_prob_to_select_parent= np.random.uniform(0,1)
print(f'uniform_prob_to_select_parent: {uniform_prob_to_select_parent}')

sum_of_list= 0
for position, value in enumerate(probability_to_select_parent):
    sum_of_list = sum_of_list + value 
    if sum_of_list >= uniform_prob_to_select_parent:
        print(f'sum_of_list: {sum_of_list}')
        print(f'position to select: {position}')
        print(sorted_list[position])
        break
    else: 
        print(f'sum_of_list: {sum_of_list}')
# repeat until lambda parents are selected.-> px 20 parents -> 20 children remove 10% thus 10 individuals are replaced by top 10 of 20 