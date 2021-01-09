# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:57:42 2021

@author: admin
"""

#Part 1- Using Self Organizing Maps to identify the frauds
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('E:/Pranav/Portfolio/Github/Deep Learning/Credit card applications using SOMs/Credit_Card_Applications.csv')

#Exploring the dataset
dataset.head()
dataset.tail()
dataset.columns
len(dataset.columns)
len(dataset)
dataset.info()
dataset.describe()

#Creating the Dependent and Independent variables 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,5)], mappings[(2,6)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#Part 2 - Building an ANN to predict the probability of comiting frauds
#Creating the Independent variable
customers = dataset.iloc[:, 1:].values

#Creating the Dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] == 1
        
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#Building the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(customers,is_fraud,batch_size = 1, epochs = 2)

#Probabilities of frauds
y_pred = ann.predict(customers)

#Adding customer id to y_pred
y_pred = np.concatenate((dataset.iloc[:,0:1],y_pred),axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]