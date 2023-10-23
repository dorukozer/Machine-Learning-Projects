#!/usr/bin/env python
# coding: utf-8

# In[178]:


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd










# In[179]:



np.random.seed(41222)
# mean parameters
class_means = np.array([[+0.0, +2.5],
                        [-2.5, -2.0],
                       [+2.5, -2.0]])
# covariance parameters
class_covariances = np.array([[[+3.2, +0.0], 
                               [+0.0, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]],
                            [[+1.2, -0.8], 
                               [-0.8, +1.2]]])
# sample sizes
class_sizes = np.array([120, 80,100])


# In[180]:


# generate random samples
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2,points3))


# generate corresponding labels
y = np.concatenate((np.repeat(0, class_sizes[0]), np.repeat(1, class_sizes[1]),np.repeat(2, class_sizes[2])))

XY =np.hstack((X, y[:, None]))


# In[181]:


# write data to a file
np.savetxt("homework1_data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")


# In[182]:


# plot data points generated
plt.figure(figsize = (8, 8))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[183]:


# read data into memory
# burdan başlıyor normalde 
 # class number kaç tane class var
# read data into memory
data_set = np.genfromtxt("homework1_data_set.csv", delimiter = ",")
   
# get X and y values
Xd = data_set[:,[0,1]]



X0 = data_set[:,[0]]
X1 = data_set[:,[1]]
y_truth = data_set[:,2].astype(int)





# get number of samples
#N = data_set.shape[0]

#print(X[0,[0,0]])

#print(X[y_truth == (0)][0])

sample_means = [np.mean(X[y_truth == (c)], axis=0) for c in range(3)]

print(sample_means)


sample_covariances = [
    np.dot(np.transpose( X[y_truth == (c )] - sample_means[c]),  X[y_truth == (c )] - sample_means[c]) / class_sizes[
        c] for c in range(3)]

class_priors= [   len(X[y_truth == (c)])/(len(X))   for c in range(3)]

print(class_priors)


# 

# In[184]:


def score(Xv,X,sample_covariances,sample_means):
    a=-999999999999999999999
    k= -1 
    for c in range(3):
        b= (-1*np.log(2*math.pi))-1/2*(np.log(np.linalg.det(sample_covariances[c])))-(1/2*np.dot(np.dot(np.linalg.inv(sample_covariances[c]),( Xv - sample_means[c])),np.transpose( Xv- sample_means[c])))
        + np.log(class_priors[c])
        if b > a:
            a=b
            k=c
           
    return k           


        
def confusionMatrix2(Xv,X,sample_covariances,sample_means):
    M= []
    
    for a in range (len(Xv)):
            
        if 0 == score(Xv[a],X,sample_covariances,sample_means):
            M.append(0) 
        if 1 == score(Xv[a],X,sample_covariances,sample_means):
             M.append(1)
        if 2 == score(Xv[a],X,sample_covariances,sample_means):
             M.append(2) 
    return(M)




    
    


# In[185]:


CMatrix1= (confusionMatrix2(X[y_truth ==0],X,sample_covariances,sample_means))
CMatrix2=(confusionMatrix2(X[y_truth ==1],X,sample_covariances,sample_means))
CMatrix3=(confusionMatrix2(X[y_truth ==2],X,sample_covariances,sample_means))
Matrix =np.concatenate((CMatrix1,CMatrix2))
Mfin = np.concatenate((Matrix, CMatrix3))

confusion_matrix = pd.crosstab(Mfin,y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[186]:


# evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)



plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 0, 0], X[y_truth == 0, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "b.", markersize = 10)
plt.plot(X[Mfin != y_truth, 0], X[Mfin != y_truth, 1], "ko", markersize = 12, fillstyle = "none")


plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




