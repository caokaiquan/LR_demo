import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.getcwd())
columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
cars = pd.read_table('auto-mpg.data',delim_whitespace=True,names = columns)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

cars.plot('weight','mpg',kind = 'scatter',ax = ax1)
cars.plot('acceleration','mpg',kind = 'scatter',ax = ax2)
plt.show()


import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(cars[['weight']],cars['mpg'])










