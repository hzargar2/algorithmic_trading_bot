import numpy as np
import pandas as pd

df = pd.DataFrame()
df['values'] = [2,3,4,5,6]
df['neg'] = [-2,-3,-4,-5,-6]
print(df.head())
values = df['values'].values.reshape(-1,1)
neg = df['neg'].values.reshape(-1,1)
print(values)
print(neg)
data = np.concatenate((values,neg), axis = 1)
print(data)
data = data.reshape(1, 5, 2)
print(data)

ls = [0,1,2,3,4,5]
print(ls[3])