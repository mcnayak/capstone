# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('train.csv')

sales = data['sales']
features_df = data.drop('sales', axis = 1)

median_sales = np.mean(features_df)
#visualize Sales per Item across stores
store1_sales = data[data['store' == 1]]
print(store1_sales.sum())
store1_sales.groupby('item')
#store1_sales.groupby('item').sum()