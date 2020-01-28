import random
import pandas as pd
import numpy as np
arr = {'a':'apple','b':'ball','c':'cat','d':'dog','e':'ele','f':'fish'}
keys = list(arr.keys())
random.shuffle(keys)
crr = [(key,arr[key]) for key in keys]
print(type(crr))
print(crr)
drr = np.array(crr)
data = drr[:,0]
target = drr[:,1]
print(data)
print(target)
# print(arr)
# brr = pd.Series(arr)
# print(brr)
# crr = np.array(arr)
# print("crr:",crr)
# print(crr.size())
# print(len(crr))
# np.random.shuffle(crr)
# print(crr)
# print(type(crr))