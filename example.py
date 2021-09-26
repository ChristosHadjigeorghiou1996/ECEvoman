import numpy as np 
ar = np.array([ -1, 5 , 3, -2, -5, 10, 0.05, -0.02])
print(f'original:{ar}')
new_ar= [x > 1 for x in ar]
ar[[x<-1 for x in ar]]= -1

ar[[x>1 for x in ar]]= 1
print(f'ar:{ar}')

# ar = ar[x in ]
a= np.random.randint(1,3+1, 1)
print(f'a:{a}')
print(f'a:{a[0]}')

for pos, x in enumerate(["a", "b", "c"]):
    print(pos, x )

mutation_probabilities= np.random.uniform(0, 1, (265))
# print(f'mutation_probabilities: {mutation_probabilities}')

