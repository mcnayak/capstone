# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import dateutil
from dateutil.parser import parse
plt.figure(figsize=(20,5))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('split/train.csv')

#Exploratory Data Analysis 
print('Exploratory Analysis',data.shape)
#print(data.describe())
#print(data['item'].count())
#print(data['store'].count())
 
sales = data['sales']
features = data.drop('sales', axis = 1)

# Group by of Sales on Store and Item
#print(data.groupby(['date','item'])['sales'].sum())

#Set Date Column as the Index
data.set_index(data['date'],inplace=True)
print('Date Index',data.index)

#Convert the data from string to Date format

data['month'] = data['date'].apply(dateutil.parser.parse, dayfirst=False)
#print(data.groupby(['month']).groups.keys())
#print(data.shape)
#print(data.head)
data['quarter'] = pd.PeriodIndex(data['month'],freq='Q')


# Group Sales by Quarter, Store , Item 
print('Data Grouped by q,s,i',data.groupby(['quarter','store','item'])['sales'].sum())

# Group Sales by Quarter, Item 
print('Data Grouped by q,i',data.groupby(['quarter','item'])['sales'].sum())
#print(data.groupby(['Quarter','item']).describe().unstack())



# ARIMA Analysis of Sales 


#Visualize Sales vs Quarter for a store
#q_sales_df = data.groupby(['quarter','item'])['sales'].sum()
import matplotlib.pyplot as plt
import statsmodels.api as sm
#store_data = data.filter(item=['6'],axis=0)
X = data['date']
y = data['sales']

#plt.plot(X,y)
#Data visualization showed a pattern in sales
#plt.show()

from pylab import rcParams
sdata = pd.read_csv('split/train.csv')
sdata.reset_index(inplace=True)
sdata['date'] = pd.to_datetime(sdata['date'])
sdata = sdata.set_index('date')
X = sdata['date']
y = sdata['sales']

rcParams['figure.figsize']=11,9
decomposition = sm.tsa.seasonal_decompose(y,model='additive')
fig = decomposition.plot()
plt.show()

#Using ARIMA model 


#features = data['date']
#plt.plot()
#for i, col in enumerate(features.columns):
    # 3 plots here hence 1, 3
#plt.subplot(1, 2, i+1)
#X = features
#y = sales
#plt.plot(x, y, 'x')
# Create regression line
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
#plt.title(sales)
#plt.xlabel(features)
#plt.ylabel('sales')

#Visualize Sales vs Quarter for all stores

#Cross Validation 
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Linear Regression of Sales
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#print('1.Regression score',regressor.score(X_train,y_train))
#y_pred = regressor.predict(X_test)
#print('1. Logistic Regression:', y_pred)





    

#Groupby Month
#data['count'].resample('M',how='sum')
#print(data.groupby(pd.PeriodIndex(pd.to_datetime(['date']),freq='Q'),axis=0).mean())

#median_sales = np.mean(features_df)
#visualize Sales per Item across stores
#store1_sales = data[data['store'] == 1]
#print(data.groupby(['item']).sum())
#print(data.groupby(['store','item']).sum())

#data['count'].resample('M', how='sum')
#data['month'] = data['date'].apply(dateutil.parser.parse, dayfirst=False)
#data.groupby(['month']).groups.keys()
#len(data.groupby(['month']).groups['08'])
#print(data.groupby(['date','item'])['sales'].sum())
#data.groupby(pd.PeriodIndex(['date'], freq='Q'), axis=0).mean()
#print(store1_sales.groupby('item').sum())s
#print(data.groupby('month','item')['date'].count())
#s2 = data.groupby([lambda x: x.month]).sum()
#data.resample('M',how=sum)
#print(s2)
#aonao = pd.DataFrame({'AO':AO.to_period(freq='M'), 'NAO':NAO.to_period(freq='M')} ) sample
#type(AO.index)
#type(AO.to_period(freq='M').index)
#q_mean = aonao.resample('Q-NOV')