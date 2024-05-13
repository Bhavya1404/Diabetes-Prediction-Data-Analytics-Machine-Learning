#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


dataset = pd.read_csv('diabetes.csv')


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


dataset.describe().T


# In[8]:


dataset.isnull().sum()


# In[48]:


dataa.duplicated().sum()


# In[9]:


sns.countplot(x = 'Outcome',data = dataset)


# In[10]:


sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()


# In[11]:


sns.heatmap(dataset.corr(), annot = True)
plt.show()


# In[13]:


dataset_new = dataset
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)


# In[14]:


dataset_new.isnull().sum()


# In[15]:


dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)


# In[16]:


dataset_new.isnull().sum()


# In[18]:


y = dataset_new['Outcome']
X = dataset_new.drop('Outcome', axis=1)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )


# In[20]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
y_predict = model.predict(X_test)


# In[21]:


y_predict


# In[22]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_predict)
cm


# In[23]:


sns.heatmap(pd.DataFrame(cm), annot=True)


# In[24]:


from sklearn.metrics import accuracy_score


# In[25]:


accuracy =accuracy_score(Y_test, y_predict)
accuracy


# In[26]:


y_predict = model.predict([[1,148,72,35,79.799,33.6,0.627,50]])
print(y_predict)
if y_predict==1:
    print("Diabetic")
else:
    print("Non Diabetic")


# In[ ]:




