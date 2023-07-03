#!/usr/bin/env python
# coding: utf-8

# # Assingment 3

# # Ritik Kumar
# # REG NO- 20BEE0019

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


dt = pd.read_csv('Housing.csv')


# In[3]:


dt


# In[4]:


dt.info()


# In[5]:


dt.head()


# In[6]:


dt.tail()


# In[8]:


dt.shape


# # Descriptive analysis

# In[9]:


dt.describe()


# In[10]:


dt.columns.values


# In[11]:


dt['mainroad'].value_counts()


# In[12]:


dt['guestroom'].value_counts()


# In[13]:


dt['hotwaterheating'].value_counts()


# In[14]:


dt['airconditioning'].value_counts()


# ## Since hotwaterheating column has very less number of "yes" values hence we would drop this column from our data as it is an imbalance data¶

# In[15]:


dt.drop(columns=['hotwaterheating'],inplace=True)


# In[16]:


#After dropping the column hotwaterheating
dt


# In[17]:


import seaborn as sns


# In[18]:


sns.histplot(x=dt['price'])


# In[19]:


sns.countplot(x=dt['bedrooms'])


# In[20]:


sns.countplot(x=dt['stories'])


# In[21]:


sns.countplot(x=dt['mainroad'])


# In[22]:


sns.countplot(x=dt['guestroom'])


# In[23]:


sns.countplot(x=dt['basement'])


# In[24]:


sns.countplot(x=dt['airconditioning'])


# In[25]:


sns.countplot(x=dt['parking'])


# In[26]:


sns.countplot(x=dt['furnishingstatus'])


# In[27]:


sns.distplot(x=dt['price'])


# In[28]:


print(dt['price'].skew())
print(dt['price'].kurt())


# In[29]:


sns.distplot(x=dt['area'])


# In[30]:


print(dt['area'].skew())
print(dt['area'].kurt())


# In[31]:


sns.boxplot(x=dt['price'])


# In[32]:


sns.boxplot(x=dt['area'])


# In[36]:


sns.heatmap(dt.corr())


# In[37]:


dt


# In[39]:


q1 = np.percentile(dt['price'],25)
q3 = np.percentile(dt['price'],75)

lowOutlier =  q1 - 1.5*(q3-q1)
highOutlier =  q3 + 1.5*(q3-q1)

dt = dt[(dt['price']>lowOutlier) & (dt['price']<highOutlier)]


# In[40]:


dt.shape


# ### Categorical columns to numerical columns using hot encoding

# In[41]:


# mainroad , guestroom , basement , airconditioning ,furnishingstatus are these columns to be transformed
dt = pd.get_dummies(data=dt , columns=['mainroad','guestroom' ,'basement','airconditioning','furnishingstatus'],drop_first=True)


# In[42]:


dt


# In[43]:


dt.shape


# In[44]:


sns.heatmap(dt.corr() , cmap = 'summer')


# In[45]:


dt['price_per_sqft'] = (dt['price']/dt['area']).astype('int64')


# In[46]:


dt


# In[47]:


sns.boxplot(x=dt['price_per_sqft'])


# In[48]:


dt = dt[(dt['price_per_sqft'])<=1700]


# In[49]:


dt.shape


# In[50]:


sns.boxplot(x=dt['price_per_sqft'])


# In[51]:


sns.boxplot(x=dt['price'])


# In[52]:


dt = dt[(dt['price'])<=8e6]


# In[53]:


dt.shape


# In[54]:


sns.boxplot(x=dt['price'])


# In[55]:


sns.boxplot(x=dt['area'])


# In[56]:


dt = dt[(dt['area'])<=9300]


# In[57]:


dt.shape


# In[58]:


sns.boxplot(x=dt['area'])


# ### Import library sklearn

# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder , StandardScaler


# In[60]:


X = dt.drop(columns=['price'])
y= dt['price']


# In[61]:


X


# In[63]:


y


# In[64]:


X_train ,X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 ,  random_state= 0 )


# In[65]:


X_train.shape


# In[66]:


X_test.shape


# In[67]:


from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[68]:


scaler = StandardScaler()


# In[69]:


lr = LinearRegression()


# In[70]:


pipe = make_pipeline(scaler,lr)


# In[71]:


pipe.fit(X_train , y_train)


# In[72]:


y_pred_lr = pipe.predict(X_test)


# In[73]:


r2_score(y_test , y_pred_lr)


# In[74]:


lasso = Lasso()


# In[75]:


pipe = make_pipeline(scaler,lasso)


# In[76]:


pipe.fit(X_train , y_train)


# In[77]:


y_pred_lasso = pipe.predict(X_test)


# In[78]:


r2_score(y_test , y_pred_lasso)


# In[79]:


ridge = Ridge()


# In[80]:


pipe = make_pipeline(scaler , ridge)


# In[81]:


pipe.fit(X_train , y_train)


# In[82]:


y_pred_ridge = pipe.predict(X_test)


# In[83]:


r2_score(y_test , y_pred_ridge)


# In[84]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[85]:


# Mean Squared Error (MSE)
MSE = mean_squared_error(y_test, y_pred_lasso)
print("Mean Squared Error:", MSE)

# Mean Absolute Error (MAE)
MSE = mean_absolute_error(y_test, y_pred_lasso)
print("Mean Absolute Error:", MSE)

# R-squared score
r2 = r2_score(y_test, y_pred_lasso)
print("R-squared Score:", r2)


# In[86]:


MSE = mean_squared_error(y_test, y_pred_lr)
print("Mean Squared Error:", MSE)

# Mean Absolute Error (MAE)
MSE = mean_absolute_error(y_test, y_pred_lr)
print("Mean Absolute Error:", MSE)

# R-squared score
r2 = r2_score(y_test, y_pred_lr)
print("R-squared Score:", r2)


# In[ ]:




