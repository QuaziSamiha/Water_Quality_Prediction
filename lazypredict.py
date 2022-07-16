import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from lazypredict.Supervised import LazyRegressor
from lazypredict import LazyRegressor

df = pd.read_csv('water_potability.csv')
df.shape