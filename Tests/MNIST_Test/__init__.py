import pickle
import numpy as np

a = np.array(np.random.rand(1000,1000,3),dtype=np.float16)

with open("pickle.pkl",'wb') as f:
    pickle.dump(a,f)
