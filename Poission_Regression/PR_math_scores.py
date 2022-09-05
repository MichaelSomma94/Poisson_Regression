# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:18:21 2022

@author: Michael
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

#

data = pd.read_csv("competition_awards_data.csv")
data = data.dropna()

print(data.head())

# figure = px.bar(data_frame = data, x="Math Score",
#                     y="Awards")
# figure.show()

x = np.array(data[['Math Score']])
y = np.array(data[['Awards']])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=1)

pipeline = Pipeline([('model', PoissonRegressor())])
pipeline.fit(xtrain, ytrain.ravel())
ypred_Poisson = pipeline.predict(xtest)
r2_test_Poisson = metrics.r2_score(ytest, ypred_Poisson)

model = LinearRegression()
model.fit(xtrain, ytrain.ravel())
ypred_Linear = model.predict(xtest)
r2_test_Linear = metrics.r2_score(ytest, ypred_Linear)
print(r2_test_Poisson,r2_test_Linear)

print(model.predict([[100]]), pipeline.predict([[100]]))

fig, ax = plt.subplots()
line1=ax.plot(xtrain, ytrain ,'bo', label='Poisson_model')
line2=ax.plot(xtest, ypred_Poisson ,'ro', label='Poisson_model') #1/radius
line3=ax.plot(xtest, ypred_Linear,'go', label='Linear model')
ax.set(xlabel=r'math score', ylabel=r'# of awards')
ax.grid()
ax.legend()
#fig.savefig("electron_doses_fit.pdf",format='pdf')
plt.show()