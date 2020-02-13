import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(len(arr))
arr = np.concatenate((np.ones((len(arr),1)),arr),axis=1)
print(arr)