import random
import pandas as pd
import numpy as np
dict= {}
arr = np.array(['d','b','c','a','d','e','b','e'])
crr = np.unique(arr)
print(crr)
# for i in range(len(crr)):
dict[crr[0]] = [1,2,3]
dict[crr[0]].append(5)
print(dict.items())






# arr = {'a':'apple','b':'ball','c':'cat','d':'dog','e':'ele','f':'fish'}
# keys = list(arr.keys())
# random.shuffle(keys)
# crr = [(key,arr[key]) for key in keys]
# print(type(crr))
# print(crr)
# drr = np.array(crr)
# data = drr[:,0]
# target = drr[:,1]
# print(data)
# print(target)
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