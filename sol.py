# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:45:33 2020

@author: Sanket Kale
"""

import numpy as np
import pandas as pd 


dataset = pd.read_csv('corona.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,10].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/5, random_state=0)


from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state=0)
cls.fit(X_train, Y_train)

Y_pred = cls.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

prob_infection =cls.predict_proba([98,0,0,0,0,1,0,0,-1,22])

