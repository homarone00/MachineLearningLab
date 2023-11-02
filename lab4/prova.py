import numpy as np

prova=np.array([1,2,3,4,5,6,7,8,9,10])
ran=np.random.choice([-1,1])
print(ran * np.where(prova>5,-1,1))