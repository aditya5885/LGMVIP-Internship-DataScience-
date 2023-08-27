#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[68]:


df=pd.read_csv('Downloads\iris.csv') #importing dataset


# In[69]:


df.head() #displays top 5 rows of dataset


# In[70]:


df.head(15) #displays top 15


# In[71]:


df.shape #total no of rows,columns


# In[72]:


df.isnull() #if false,it means no null value exists 


# In[73]:


df.dtypes #datatypes


# In[74]:


df.groupby('species')      #grouping by species


# In[75]:


df['species'].unique()  #provides unique types 


# In[76]:


df.info() #gives info


# In[77]:


pt.boxplot(df['sepal_length']) #box plot of 'sepal_length'. 


# In[78]:


#visualize whole dataset
sn.pairplot(df) #visualize the relationships between multiple pairs of columns


# In[79]:


df["species"].value_counts() #counts total of each type or species


# In[80]:


sn.FacetGrid(df ,hue="species",height=8).map(pt.scatter,"petal_length","sepal_length") ## a scatter plot with different colors representing different species in the "species" 


# In[81]:


sn.boxplot(x='species',y='petal_length',data=df)
pt.show()   #distribution of "petal_length" variable across different species in your DataFrame 


# In[82]:


sn.boxplot(x='species',y='sepal_length',data=df)
pt.show()


# In[83]:


pt.figure(figsize=(10,7))
sn.heatmap(df.corr(),annot=True,cmap="seismic")
pt.show() # heatmap that visualizes the correlation matrix of your DataFrame 


# In[84]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()


# In[85]:


df['species']=le.fit_transform(df['species'])
df.head()   #encode the categorical variable 'species' into numerical values


# In[86]:


# Print the accuracy scores of all models
for model, score in zip(models, scores):
    print(f"Accuracy of {type(model).__name__}: {score:.4f}")

# Find the best performing model
best_model_index = np.argmax(scores)
best_model = models[best_model_index]
best_model_name = type(best_model).__name__
best_accuracy = scores[best_model_index]
print(f"\nBest performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




