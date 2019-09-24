#!/usr/bin/env python
# coding: utf-8

# In[11]:


x = np.array([100,120,150,170,200,200,202,203,205,210,215,250,270,300,305,310])
y = np.array([1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0])

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x,y)
plt.title("Pricing Bids")
plt.xlabel('Price')
plt.ylabel('Status (1:Won, 0:Lost)')

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')

#Convert a 1D array to a 2D array in numpy
X = x.reshape(-1,1)
#Run Logistic Regression
logreg.fit(X, y)

logreg.predict(X)
#array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(logreg.predict([[100]]))
print(logreg.predict([[300]]))


# In[12]:



#predict prices
prices = np.arange(180, 230, 0.5)
probabilities= []

#plot in a new cell 
for i in prices:
    p_loss, p_win = logreg.predict_proba([[i]])[0]
    probabilities.append(p_win)
   
plt.scatter(prices,probabilities)
plt.title("Logistic Regression Model")
plt.xlabel('Price')
plt.ylabel('Status (1:Won, 0:Lost)')

