#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math 
def safelog(x):
    return(np.log(x + 1e-100))



# In[24]:


# read data into memory
data_set = np.genfromtxt("hw02_images.csv", delimiter = ",")


# In[25]:


X= np.zeros((30000,784))
Xtested= np.zeros((5000,784))

for i in range( 30000):
    X[i]= data_set[i]
    
for i in range(5000):
    Xtested[i]= data_set[30000+i]
    
    


# In[26]:


print(X.shape)


# In[27]:


y = np.genfromtxt("hw02_labels.csv", delimiter = ",").astype(int)


# In[28]:


y_t= np.zeros((30000))
y_t_tested= np.zeros((5000))
for i in range( 30000):
    y_t[i]= y[i].astype(int)
for i in range( 5000):
    y_t_tested[i]= y[30000+i].astype(int)
print(y_t.astype(int))
y_truth =y_t.astype(int)


# In[29]:


sample_means = np.array(([np.mean(X[y_truth == (c + 1)],axis=0) for c in range(5)]))
sample_deviations =np.array(( [np.sqrt(np.mean((X[y_truth == (c + 1)] - sample_means[c])**2,axis=0)) for c in range(5)]))
class_priors = [np.mean(y_truth == (c + 1),axis=0) for c in range(5)]
print("sample means :" , sample_means)
print("sample deviations : ", sample_deviations)
print("class priors : ", class_priors)


# In[30]:


def gi(sample_means,class_priors,sample_deviations,x):
    arrayim=np.zeros((5,784))
    for c in range(5): 
        k=np.array(-np.log(sample_deviations[c])-((1/2)*np.log(math.pi )) - 0.5 * ((x- sample_means[c])**2 /sample_deviations[c]**2) + class_priors[c]   )              
        arrayim[c]=k
        
    return(arrayim)
    
    





# In[31]:



def find_score(a):
    arrayim=np.zeros(5)
    for j in range(5):
        b=0
        for i in range(784):
            b= b+a[j][i]
        arrayim[j]=b
    
    current_score=-99999999999999999999999
    maxindex=0
    for i in range(5):
        if arrayim[i] > current_score:
            maxindex=i+1
            current_score= arrayim[i]
    return(maxindex)
a=gi(sample_means,class_priors,sample_deviations,X[29])
    
#gi(sample_means,class_priors,sample_deviations,X[0])


# In[32]:


score_predicted=np.zeros(30000) 
index_predicted=0
for k in range(30000):
    score_predicted[index_predicted]=find_score(gi(sample_means,class_priors,sample_deviations,X[k]))
    index_predicted=1 +  index_predicted
print(score_predicted)
    


# In[33]:


confusion_matrix = pd.crosstab(score_predicted.astype(int),y_t.astype(int), rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[34]:


score_predicted=np.zeros(5000) 
index_predicted=0
for k in range(5000):
    score_predicted[index_predicted]=find_score(gi(sample_means,class_priors,sample_deviations,Xtested[k]))
    index_predicted=1 +  index_predicted
confusion_matrix = pd.crosstab(score_predicted.astype(int),y_t_tested.astype(int), rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)
    


# In[ ]:





# In[ ]:




