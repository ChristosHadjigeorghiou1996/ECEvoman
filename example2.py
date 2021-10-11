# logarithm for changing sigma which has normal distribution
import numpy as np
from numpy import array
from math import sqrt, exp

probability_lower_bound = 0
probability_upper_bound = 1 

combined_list_population_individual_fitness= [[array([-0.59396292,  0.17895778,  0.37560988, -0.39420565,  0.82967795,
       -0.54642452,  0.45997766,  0.87455079, -0.93652417,  0.09274601,
        0.18863918,  0.68805717, -0.8302697 ,  0.63264438,  0.96179378,
       -0.23015375, -0.27462302, -0.02270719,  0.24139686, -0.0521071 ,
        0.09543217,  0.47778001, -0.26975566, -0.49277244,  0.69356146,
       -0.05998195,  0.70737304, -0.06642954, -0.9484283 , -0.77785318,
        0.58254021,  0.10627586, -0.52753073, -0.14809329, -0.85914067,
        0.43866131,  0.21898928, -0.76767913,  0.151932  , -0.86858737,
       -0.80731398,  0.28659804,  0.56093264,  0.99772533, -0.09554562,
        0.35760855, -0.07214945, -0.42961878,  0.72840675,  0.9037848 ,
       -0.59531179,  0.37577466,  0.80141909, -0.86089334,  0.55427584,
        0.96340382,  0.39515026,  0.29448894, -0.75267415,  0.14784414,
        0.30045862, -0.93859847,  0.3320083 ,  0.11844599, -0.84486494,
        0.19652789,  0.30598785,  0.90941583, -0.09309091, -0.26826885,
       -0.3542321 , -0.34731536, -0.22156885,  0.25304174, -0.35295317,
        0.35871251, -0.1404071 , -0.12327236,  0.39469004, -0.29667455,
       -0.27813094, -0.44390774, -0.58788214, -0.5158528 , -0.88297678,
        0.3471684 , -0.12012875, -0.90034768,  0.04354566,  0.9948859 ,
       -0.71059742, -0.10936027, -0.23259531,  0.07764255,  0.93695878,
       -0.71773752,  0.26077156, -0.16094614,  0.49617194, -0.39807091,
        0.86634955, -0.03083974, -0.50697909,  0.67593286, -0.67754633,
        0.37848238, -0.37289074, -0.4060573 ,  0.68011575,  0.24967066,
       -0.66005217, -0.21324316,  0.58229957,  0.09711935,  0.56830617,
       -0.33798722,  0.09107815,  0.78630672, -0.37435927, -0.26025353,
       -0.71733493, -0.08535639, -0.45957261,  0.03043816, -0.542144  ,
        0.85552482, -0.7481791 ,  0.7460297 , -0.2806774 , -0.77543083,
        0.52533357,  0.31495413, -0.94797303,  0.38043539, -0.98884754,
        0.49427612,  0.5152994 ,  0.51614759,  0.12124443, -0.11740407,
        0.66515264,  0.95858428, -0.02909649, -0.56010783,  0.97612172,
       -0.91497313,  0.52483605,  0.49846472, -0.34263709, -0.57882211,
        0.44245785,  0.69430917, -0.44659255, -0.4795174 , -0.70491158,
        0.26111932, -0.79001563,  0.01601289, -0.9018299 ,  0.00607738,
       -0.88896785,  0.53147169,  0.97568486,  0.18333173, -0.68616693,
       -0.21292667,  0.66515006,  0.51811916,  0.0062821 ,  0.43419633,
       -0.16078856,  0.63954615, -0.91312861,  0.70990379,  0.50440996,
        0.08507056,  0.80002037, -0.90796061,  0.23080479, -0.79285905,
       -0.08244237, -0.44537818, -0.73242596,  0.76086725,  0.12851965,
       -0.63748822, -0.62872572, -0.16725265,  0.09916649, -0.97711108,
        0.66873764, -0.76326901, -0.7615432 ,  0.2512177 ,  0.68734625,
        0.57305573, -0.82103555, -0.11955988, -0.16862199, -0.99047816,
       -0.67353004, -0.96662892,  0.75441492,  0.13189996, -0.64017396,
       -0.11772849,  0.54003417,  0.8491676 , -0.39023858,  0.71465467,
       -0.87879458,  0.59168397, -0.74719952, -0.31455335,  0.18941409,
        0.0773094 , -0.18750859,  0.94266664,  0.6596574 ,  0.7740319 ,
        0.4978338 , -0.72356206,  0.84214496, -0.95507322,  0.42268647,
       -0.14449725, -0.81855866, -0.84478617, -0.72956927, -0.06001695,
        0.70161426,  0.99887754,  0.36306509,  0.07123606, -0.41772635,
        0.17272147,  0.82316461,  0.24935923, -0.13357195, -0.33333483,
       -0.01688602, -0.45936138,  0.2948778 ,  0.97578077,  0.99756706,
        0.10722735, -0.38059756, -0.96423957,  0.35721036, -0.65786976,
       -0.09990772,  0.90867909, -0.54904628, -0.06717925,  0.41137644,
       -0.22287551,  0.26202422, -0.98911375,  0.40616923, -0.57649166,
       -0.78056675,  0.02492741, -0.59975868, -0.61508873,  0.34696029,]), 21.779644174921675], 
       [array([ 0.52089972,  0.75603559, -0.70633891, -0.73865296,  0.16798309,
        0.90590179,  0.48054099,  0.09318811, -0.81754141,  0.31946896,
       -0.30759485, -0.17187358, -0.82167449, -0.32058411, -0.3230962 ,
        0.27284627, -0.11231288,  0.75422799, -0.15168215,  0.68369326,
        0.9626329 ,  0.28259165, -0.45208664, -0.71820314, -0.76690137,
       -0.73152143,  0.81687978,  0.39585582,  0.63092292,  0.21060766,
       -0.89737314, -0.03650855, -0.73664587,  0.99069153, -0.00128677,
        0.33848062, -0.09465617, -0.21616084, -0.16480633,  0.08704748,
       -0.6543246 ,  0.43137134, -0.73538505, -0.40953514, -0.3367931 ,
       -0.31487451,  0.01410057,  0.65267822, -0.97721989,  0.29030871,
       -0.22476004,  0.92992867, -0.77482275,  0.82450653, -0.17561016,
        0.9670368 , -0.30434048,  0.71657205,  0.88374577, -0.98221758,
        0.56391561, -0.31288748, -0.77919988, -0.78826259,  0.71826021,
       -0.95980748, -0.66699496,  0.25602208, -0.72419265,  0.63112294,
        0.50282405,  0.70733275,  0.58733713, -0.30122297,  0.20024716,
        0.66109654, -0.84611157,  0.4254146 , -0.11092501,  0.18578074,
       -0.22345494, -0.63465334, -0.39611887,  0.0538132 , -0.78814757,
        0.02882197, -0.16549541,  0.72886131,  0.02935732, -0.08395929,
        0.56726688, -0.02674564, -0.00706603,  0.22738673, -0.62867947,
       -0.26787577, -0.89702848,  0.02452555, -0.48840107,  0.62551353,
        0.21706762,  0.6365207 ,  0.20172665, -0.20657008,  0.62579626,
       -0.35478899, -0.27590981,  0.37732557, -0.73589293,  0.07921731,
       -0.46406363,  0.64332545, -0.18130966,  0.24375726, -0.82412305,
        0.09492888, -0.89757893,  0.68426998,  0.29770274,  0.02306757,
        0.26139408, -0.88288146,  0.23364571, -0.93414239,  0.43545821,
       -0.75383694, -0.07795319,  0.23010409, -0.82939337,  0.81118359,
       -0.05547641,  0.55499382,  0.23740513, -0.18829309, -0.7921454 ,
        0.5513086 , -0.69717943, -0.98331102,  0.40830403, -0.10249184,
        0.99659174,  0.35098593, -0.95532876,  0.28745308,  0.44843214,
        0.67825684, -0.02319148,  0.99712263, -0.42464497, -0.98686789,
        0.98855468, -0.51528559, -0.1975202 ,  0.07715165, -0.33334861,
        0.45693965,  0.68383744,  0.20621042, -0.49374401, -0.83005565,
        0.44934456, -0.89939417,  0.47111373,  0.64728236, -0.09555162,
       -0.52550841, -0.1917592 ,  0.38018466, -0.81942186, -0.56818431,
        0.3607413 ,  0.84150124, -0.98207114,  0.79327854,  0.23053994,
        0.14122097,  0.37739432,  0.37084459,  0.26181416, -0.25557084,
        0.76725575, -0.64171453,  0.82898878,  0.9820223 , -0.05532487,
       -0.56851128, -0.99664664, -0.11375357, -0.17339878, -0.60115644,
       -0.07265091, -0.19697127, -0.21843675,  0.23986669,  0.63783658,
       -0.44196071, -0.08499084, -0.70058718,  0.84898581,  0.02106465,
       -0.81960072,  0.66171581, -0.08141663, -0.02382252, -0.89893974,
       -0.71753054, -0.38996919,  0.98773161, -0.50061488, -0.96117541,
        0.92779065, -0.1696344 ,  0.41223919,  0.30754034,  0.2337291 ,
       -0.42470958, -0.11483933, -0.52398183,  0.05882778,  0.77344316,
        0.48230397,  0.13177702, -0.69099024,  0.57927428,  0.38457207,
        0.67913863,  0.2582514 ,  0.83805059,  0.20697223,  0.05692856,
       -0.46320003, -0.99488002,  0.21819882,  0.56679796,  0.24431889,
        0.41597541,  0.06574921, -0.85452304, -0.47006731,  0.70382067,
        0.43515951,  0.39851585,  0.13072894,  0.97677034, -0.11248756,
        0.8217724 , -0.51155617, -0.00196679, -0.23740107,  0.89026188,
       -0.49034177, -0.18315811, -0.42470278,  0.70182471, -0.07685557,
        0.35771069,  0.4525506 ,  0.61504248, -0.07893261,  0.27896326,
       -0.78292137,  0.43565198, -0.13543668, -0.93170053, -0.26106575]), -5.814130531825066]]

def uncorrelated_mutation_with_one_sigma(individual, probability_of_mutation, current_generation, total_number_of_generations):
    # individual is 265 + 1 
    # each individual coordinate is mutated with a different noise as you draw a new value 
    # value of t is recommended from the literature **cite
    # boundary threshold of too close to 0 then push it to the boundary **cite
    # initial sigma chosen to be 1 
    boundary_threshold = 0.001
    
    learning_rate_t = 1 / (sqrt(individual.shape[0] - 1))
    print(f'learning_rate_t: {learning_rate_t}')
    # get the sigma of the individual which is the last value 
    sigma= individual[individual.shape[0] - 1]
    new_sigma_for_individual = sigma *exp(learning_rate_t * np.random.normal(probability_lower_bound, probability_upper_bound))
    print(f'new_sigma_for_individual  {new_sigma_for_individual} and boundary_threshold {boundary_threshold}')
    if new_sigma_for_individual < boundary_threshold:
        # print(f'new_sigma_for_individual  {new_sigma_for_individual} is less than {boundary_threshold}')
        new_sigma_for_individual = boundary_threshold
        print(f'new_sigma after boundary threshold {new_sigma_for_individual}')
    # create an np array to pre-compute the probabilities regarding mutation probability per coordinate of individual
    random_uniform_mutation_probability_array= np.random.uniform(probability_lower_bound, probability_upper_bound, individual.shape[0] - 1)
    # create an np array to pre-compute the noise drawn from normal distribution per coordinate
    random_normal_noise_per_coordinate_array= np.random.normal(probability_upper_bound, probability_upper_bound, individual.shape[0] - 1)
    
    # last value is sigma 
    for individual_coordinate_position in range(individual.shape[0] - 1):
        # print(f'individual.shape[0] - 1): {individual.shape[0] - 1}')
        # print(f'random_uniform_mutation_probability_array[individual_coordinate_position]: {random_uniform_mutation_probability_array[individual_coordinate_position]} and probability_of_mutation: {probability_of_mutation}')
        if random_uniform_mutation_probability_array[individual_coordinate_position] < probability_of_mutation:
            print(f'before individual[individual_coordinate_position]: {individual[individual_coordinate_position]}')
            print(f'random_normal_noise_per_coordinate_array[individual_coordinate_position] : {random_normal_noise_per_coordinate_array[individual_coordinate_position] }')
            new_coordinate_value= individual[individual_coordinate_position] + new_sigma_for_individual *  random_normal_noise_per_coordinate_array[individual_coordinate_position]
            # print(f'individual[individual_coordinate_position]: {individual[individual_coordinate_position]}')
            # print(f'new_coordinate_value: {new_coordinate_value}')
            # ensure is between lower bound and upper bound
            if new_coordinate_value > probability_upper_bound:
                new_coordinate_value= probability_upper_bound
            elif new_coordinate_value < -1:
                new_coordinate_value = -1
            individual[individual_coordinate_position] = new_coordinate_value
            print(f'after individual[individual_coordinate_position]: {individual[individual_coordinate_position]}')

    individual[individual.shape[0] - 1] = new_sigma_for_individual 
    print(f'individual with new sigma: {individual}')
    return individual 

if __name__ == '__main__':
    print(type(combined_list_population_individual_fitness))
    # for ind in combined_list_population_individual_fitness:
    #     ind_value= ind[0]
    #     print(f'ind_value: {ind_value} and shape: {ind_value.shape}')
    #     modified_ind_value= ind[0][:-1]
    #     print(f'modified_ind_value: {modified_ind_value} and shape: {modified_ind_value.shape}')
    #     ind_fitness= ind[1]
    #     print(f'ind_fitness: {ind_fitness}')
    #     mutated_individual= uncorrelated_mutation_with_one_sigma(ind_value, 0.25, 1, 20)
        

