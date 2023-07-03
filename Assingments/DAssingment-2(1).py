#!/usr/bin/env python
# coding: utf-8

# # Assingment-2
# # Ritik Kumar 20BEE0019
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('titanic.csv')


# In[3]:


df


# In[4]:


df.info()


# In[48]:


df.describe()


# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[50]:


df.shape


# In[51]:


df.head()


# In[52]:


df.columns.values


# # categorical columns 
# # sex
# # class who
# # adult_male
# # deck
# # class
# # embarked
# # embark_town
# # alone
# 
# 
# # Numerical Columns
# # age
# # survived
# # pclass
# # age
# # sibsp
# # parch
# # fare

# In[53]:


# missing values in age , embark , deck , embark_town
# more than 70% values are missing in deck column
# few columns have inappropraite data type


# In[55]:


df.drop(columns=['deck'],inplace=True)


# In[58]:


df.info()


# In[59]:


# imputing missing values for age
df['age'].fillna(df['age'].mean() , inplace = True)


# In[61]:


df.info()


# In[62]:


# handling missing values for embarked
df['embarked'].fillna('S' , inplace = True)


# In[63]:


df.info()


# In[68]:


# handling missing values for embark_town
df['embark_town'].fillna('Southampton' , inplace = True)


# In[69]:


df.info()


# In[70]:


df['parch'].value_counts()


# In[71]:


df['sibsp'].value_counts()


# In[75]:


# changing datatype of the following columns
# 
df['age'] = df['age'].astype(int)
df['sex'] = df['sex'].astype('category')


# In[76]:


df.info()


# In[83]:


# there are certain columns that are repeating
# survived and alive hence remove the alive column
# embark and embark_town are same 
df.drop(columns=['embark_town'],inplace=True)
df.drop(columns=['class'],inplace=True)


# In[81]:


df


# In[84]:


df.drop(columns=['who'],inplace=True)


# In[86]:


df.drop(columns=['alone'],inplace=True)


# In[88]:


df.drop(columns=['class'],inplace=True)


# In[90]:


df.drop(columns=['adult_male'],inplace=True)


# In[92]:


df.info()


# In[93]:


df['survived'] = df['survived'].astype('category')
df['pclass'] = df['pclass'].astype('category')
df['embarked'] = df['embarked'].astype('category')


# In[94]:


df.info()


# In[95]:


df


# In[97]:


# now that we have removed unwanted and duplicate columns from the data 
# now our file size of titanic data set has reduced to 28mb 
# decreased about 1/3 
df.describe()


# In[98]:


# univariate analysis
sns.countplot(df['survived'])


# In[99]:


sns.countplot(df['pclass'])


# In[100]:


sns.countplot(df['sex'])


# In[101]:


sns.histplot(df['age'])


# In[102]:


sns.countplot(df['sibsp'])


# In[103]:


sns.countplot(df['parch'])


# In[104]:


sns.histplot(df['fare'])


# In[105]:


sns.countplot(df['embarked'])


# In[106]:


sns.distplot(df['age'])
print(df['age'].skew())
print(df['age'].kurt())


# In[107]:


sns.boxplot(df.age)


# In[110]:


print("people with age in between 60 and 70 are " , df[(df['age']>60)&(df['age']<70)].shape[0])
print("people with age in between 70 and 75 are " , df[(df['age']>=70)&(df['age']<=75)].shape[0])
print("people with age in above 75 are " , df[df['age']>75].shape[0] )
print('_'*50)
print("people with age in between 0 and 1 are " , df[df['age']<1].shape[0] )



# In[112]:


sns.distplot(df['fare'])


# In[113]:


df['fare'].skew()


# In[114]:


df['fare'].kurt()


# In[115]:


sns.boxplot(df.fare)


# In[116]:


# fare column is highly left skewed data 
# hence this column might contain outliers


# In[117]:


# multivariate analysis


# In[118]:


sns.countplot(df.survived , hue = df.pclass)


# In[119]:


sns.countplot(df.survived, hue = df.sex)


# In[120]:


sns.countplot(df.survived , hue = df.embarked)


# In[121]:


sns.distplot(df[df['survived']==0]['age'])
sns.distplot(df[df['survived']==1]['age'])


# In[122]:


sns.distplot(df[df['survived']==0]['fare'])
sns.distplot(df[df['survived']==1]['fare'])


# In[123]:


sns.pairplot(df)


# In[124]:


sns.heatmap(df.corr())


# In[125]:


df['family_size'] = df['parch'] + df['sibsp']


# In[126]:


df


# In[127]:


def family_type(number) :
    if number == 0 :
        return 'alone' 
    elif number>0 and number<=4 :
        return 'medium'
    else:
        return 'large'
    


# In[128]:


df['family_type'] = df['family_size'] . apply(family_type)


# In[129]:


df


# In[136]:


df


# In[137]:


sns.countplot(df.survived , hue = df.family_type)


# In[139]:


# handling outliers in age column 
df = df[df['age']<(df['age'].mean()*3*df['age'].std())]


# In[141]:


df.shape


# In[142]:


# handling outliers in fare column
q1 = np.percentile(df['fare'],25)
q3 = np.percentile(df['fare'],75)

outlier_low =  q1 - 1.5*(q3-q1)
outlier_high =  q3 + 1.5*(q3-q1)

df = df[(df['fare']>outlier_low) & (df['fare']<outlier_high)]


# In[143]:


df.shape


# In[147]:


#one hot encoding

# columns to be transformed are pclass , sex , embarked , family_type
df = pd.get_dummies(data=df , columns=['pclass' ,'sex' ,'embarked' ,'family_type'],drop_first=True)


# In[148]:


sns.heatmap(df.corr() , cmap  = 'summer')


# In[149]:


df


# In[150]:


# conclusion drawn 
# 1. chance of a female surviving the accident is higher than the male
# 2. pclass3 is dangerours becuase there are more dead people in this class
# 3. people embarked at C survived more
# 4 people travelling with medium family size had higher chance of survival than alone and larger family size


# In[153]:


#train_test_split
from sklearn.model_selection import train_test_split


# In[155]:


X = df.drop("survived", axis=1) 
y = df["survived"]                

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[156]:


X_train


# In[157]:


y_train


# In[158]:


X_test


# In[159]:


y_test


# In[ ]:




