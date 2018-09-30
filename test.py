import numpy as np
import pandas as pd
rng = np.random.RandomState(0)
df = pd.DataFrame({'keys': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['keys', 'data1', 'data2'])
print(df)
data = df['keys'] == "A"
print(data)
data_A = df[df['keys'] == "A"]
print(data_A)
