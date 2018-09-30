# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import dateutil
from dateutil.parser import parse
from statsmodels.tsa.arima_model import ARIMA
plt.figure(figsize=(20,5))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('split/train.csv',low_memory=False)#,parse_dates=['date'])
sample_sub = pd.read_csv('sample_submission.csv')

#Exploratory Data Analysis 
print('Exploratory Analysis',train.shape)
print(train.describe())
#print(train['item'].count())
#print(train['store'].count())
 
sales = train['sales']
features = train.drop('sales', axis = 1)

# Group by of Sales on Store and Item
print(train.groupby(['date','item'])['sales'].sum())

#Set Date Column as the Index
train.set_index(train['date'],inplace=True)
print('Date Index',train.index)

#Convert the data from string to Date format
#train['month'] = train['date'].apply(dateutil.parser.parse, dayfirst=False)
#print(train.groupby(['month']).groups.keys())
#print(train.shape)
#print(train.head)
#train['quarter'] = pd.PeriodIndex(train['month'],freq='Q')


# Group Sales by Quarter, Store , Item 
#print('train Grouped by q,s,i',train.groupby(['quarter','store','item'])['sales'].sum())

# Group Sales by Quarter, Item 
#print('train Grouped by q,i',train.groupby(['quarter','item'])['sales'].sum())
#print(train.groupby(['Quarter','item']).describe().unstack())

#Visualize Sales vs Quarter for a store
#q_sales_df = train.groupby(['quarter','item'])['sales'].sum()
import matplotlib.pyplot as plt
import statsmodels.api as sm
#store_train = train.filter(item=['6'],axis=0)
X = train['date']
y = train['sales']
plt.plot(X,y)
#train visualization showed a pattern in sales
plt.show()

#Plot the data for one Store
item_store = train[(train.item==1)&(train.store==1)].copy()

#Visualize the trend 
res = sm.tsa.seasonal_decompose(item_store.sales.dropna(),freq=365)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()

# Simple Prediction - Mean of Sales 
"""def simple_predict(train,submission):
    for _,row in train.iterrows():
        item, store = row['item'],row['store']
        day, month = row.name.day, row.name.month
        itemandstore = (train.item == item) & (train.store == store)
        dayandmonth = (train.index.month == month) & (train.index.day == day)
        train_rows = train.loc[itemandstore & dayandmonth]
        predict_sales = (train_rows.mean()['sales'])
        submission.at[row[id],'sales'] = predict_sales
    return submission

simple_pred = simple_predict(train,sample_sub.copy())
print(simple_pred)"""

""" #Using the code from DigitalOcean for trend visualization
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
plt.show() """

#Using ARIMA model 
""" def evaluate_arima_model(X,arima_order):
    # prepare training dataset
    train=X
    test = X_hat
    history = [x for x in train]
    #
p = d = q = range(0,2)
pdq = list(itertools.product(p,d,q))
seasonal_pdq =[()] """

#features = train['date']
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
#train['count'].resample('M',how='sum')
#print(train.groupby(pd.PeriodIndex(pd.to_datetime(['date']),freq='Q'),axis=0).mean())

#median_sales = np.mean(features_df)
#visualize Sales per Item across stores
#store1_sales = train[train['store'] == 1]
#print(train.groupby(['item']).sum())
#print(train.groupby(['store','item']).sum())

#train['count'].resample('M', how='sum')
#train['month'] = train['date'].apply(dateutil.parser.parse, dayfirst=False)
#train.groupby(['month']).groups.keys()
#len(train.groupby(['month']).groups['08'])
#print(train.groupby(['date','item'])['sales'].sum())
#train.groupby(pd.PeriodIndex(['date'], freq='Q'), axis=0).mean()
#print(store1_sales.groupby('item').sum())s
#print(train.groupby('month','item')['date'].count())
#s2 = train.groupby([lambda x: x.month]).sum()
#train.resample('M',how=sum)
#print(s2)
#aonao = pd.DataFrame({'AO':AO.to_period(freq='M'), 'NAO':NAO.to_period(freq='M')} ) sample
#type(AO.index)
#type(AO.to_period(freq='M').index)
#q_mean = aonao.resample('Q-NOV')