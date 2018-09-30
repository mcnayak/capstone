#Display Data 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import dateutil
from dateutil.parser import parse
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm

#train = pd.read_csv('split/train.csv',low_memory=False,parse_dates=['date'])
train = pd.read_csv('split/train.csv',low_memory=False,parse_dates=['date'],index_col=['date'])
#print(train.describe())
sample_sub = pd.read_csv('sample_submission.csv')

#Plot the data for one Store
item_store = train[(train.item==1)&(train.store==1)].copy()
print(item_store.describe())

#Visualize the trend 
res = sm.tsa.seasonal_decompose(train.sales.dropna(),freq=365)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()

items_stores = []
for store_count in range(1,2):
    for item_count in range(1,2):
        df = train[(train.item==1)&(train.store==1)].copy()
        print('df describe')
        

    

plt.figure(figsize=(20,5))