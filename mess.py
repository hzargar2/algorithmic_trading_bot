import numpy as np
import pandas as pd

lst = [[1],[2],[3],[4]]
lst2 = [[5],[6],[7],[8]]
lst3 =[]
lst3.extend([lst, lst2])
print(lst3)

lst = np.array(lst).reshape(-1,1)
lst2 = np.array(lst2).reshape(-1,1)
print(lst, lst2)

lst4 = np.concatenate((lst, lst2), axis=1)
print(lst4)

lst4 = np.array(lst4).reshape(4,2)
print(lst4)


list = ['helllllllllllllllllllllllllllllllllllllllllllpoooooooooooooooooooooooooooooooooooooooooooooooooooooooo', 'jjjjjjjjjjj']
print(list)

df = pd.DataFrame()

x = [3,4,5,6,7]
df['x'] = pd.Series(x)
y = ['a','s','d','f','g']
df['y'] = pd.Series(y)

df = df.as_matrix()
print(df)

data = []

data.append(df[:2])
print(data)
data.append(df[2:])
print(data)

