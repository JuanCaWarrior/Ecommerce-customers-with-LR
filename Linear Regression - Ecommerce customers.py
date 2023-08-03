#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv('Ecommerce Customers')
customers.head()


# In[3]:


customers.describe()


# In[4]:


customers.info()


# In[5]:


sns.pairplot(customers)


# In[6]:


customers.columns


# In[7]:


y = customers['Yearly Amount Spent']


# In[8]:


X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lm = LinearRegression()


# In[17]:


lm.fit(X_train,y_train)


# In[18]:


lm.coef_


# In[19]:


predictions = lm.predict(X_test)


# In[20]:


predictions


# In[21]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')


# In[22]:


from sklearn import metrics


# In[24]:


print('MAE', metrics.mean_absolute_error(y_test,predictions))
print('MSE', metrics.mean_squared_error(y_test,predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('EVS', metrics.explained_variance_score(y_test,predictions))


# In[25]:


sns.displot((y_test-predictions), bins=50)


# In[26]:


pd.DataFrame(lm.coef_,X.columns, columns = ['Coeff'])


# In[ ]:




