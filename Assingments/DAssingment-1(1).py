#!/usr/bin/env python
# coding: utf-8

# #                           Assingment-1
#                 
# ##                      Ritik Kumar         20BEE0019

# #### Q1. Assign your Name to variable name and Age to variable age. Make a Python program that  prints your name and age.

# In[10]:


Name=input("Enter Your name : ")
age=int(input("Enter Your age : "))
print("My name is : ",Name)
print("my age is : ",age)


# #### Q2. X="Datascience is used to extract meaningful insights." Split the string

# In[11]:


s="Datascience is used to extract meaningful insights."
s.split(" ")


# #### Q3. Make a function that gives multiplication of two numbers

# In[12]:


def multiplication(a,b):
   return a*b

multiplication(5,3)


# #### Q4. Create a Dictionary of 5 States with their capitals. also print the keys and values

# In[5]:


Dict={"Bihar":"patna","Jhkharand":"Ranchi","Rajasthan":"Jaipur","TamilNadu":"Chennai","Karnatka":"Bengaluru"}
list(Dict.keys())


# In[6]:


list(Dict.values())


# #### Q5. Create a list of 1000 numbers using range function

# In[25]:


number=list(range(1000))
print(number)


# #### Q6. Create an identity matrix of dimension 4 by 4

# In[21]:


import numpy as np
identity_matrix=np.eye(4)
print(identity_matrix)


# #### Q7. Create a 3x3 matrix with values ranging from 1 to 9

# In[24]:


import numpy as np
np.arange(1,10).reshape(3,3)


# #### Q8. Create 2 similar dimensional array and perform sum on them.

# In[28]:


a=np.array([[1,2,3,],[2,3,4],[4,5,6]])
b=np.array([[2,3,4],[1,2,3],[7,4,6]])
print("a+b= ",a+b)


# #### Q9. Generate the series of dates from 1st Feb, 2023 to 1st March, 2023 (both inclusive)

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
start_date = datetime(2023, 2, 1)
end_date = datetime(2023, 3, 1)
curr_date = start_date
while curr_date <= end_date:
    print(curr_date.strftime("%Y-%m-%d"))
    curr_date += timedelta(days=1)


# ####  Q10. Given a dictionary, convert it into corresponding dataframe and display it dictionary = {'Brand': ['Maruti', 'Renault', 'Hyndai'], 'Sales' : [250, 200, 240]

# In[35]:


dictionary = {'Brand': ['Maruti', 'Renault', 'Hyndai'], 'Sales' : [250, 200, 240]}
print(dictionary)


# In[37]:


data=pd.DataFrame(dictionary);
data


# In[ ]:




