#!/usr/bin/env python
# coding: utf-8

# In[221]:


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


# In[222]:



np.random.seed(910)
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


# In[223]:


points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2,points3))
N=300

y_truth = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]),np.repeat(3, class_sizes[2])))
K = np.max(y_truth)
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1
print(Y_truth[:,1].shape)


# In[224]:


plt.figure(figsize = (8, 8))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[225]:


eta = 0.01
epsilon = 0.001
# randomly initalize W and w0
W = np.random.uniform(low = -0.01, high = 0.01, size = (X.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))
N= X.shape[0]
print(w0)
print(np.vstack((W, w0)).shape)


# In[226]:


sample_means = np.array(([np.mean(X[y_truth == (c+1 )],axis=0) for c in range(3)]))
sample_deviations =np.array(( [np.sqrt(np.mean((X[y_truth == (c+1)] - sample_means[c])**2,axis=0)) for c in range(3)]))
class_priors = [np.mean(y_truth == (c+1),axis=0) for c in range(K)]
print("sample means :" , -sample_means)
print("sample deviations : ", sample_deviations)
print("class priors : ", class_priors)


# In[227]:


# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# In[228]:


# define the gradient functions



def gradient_w(X, y_truth, y_predicted):
    return(-np.transpose(np.dot(np.transpose((y_truth - y_predicted)*y_predicted*(1-y_predicted)),X)))

def gradient_w0(y_truth, y_predicted):
    return(-np.sum((y_truth - y_predicted)*y_predicted*(1-y_predicted)))


def gradient_W0c(X, Y_truth, Y_predicted,W,w0):
    a=np.zeros([])
    b=np.array((3,2))
    
    c=0
    a=np.zeros([])
    for i in range(N):
        wx=(X[i][0]*W[0,c]+ X[i][1]*W[1,c])  
        partial=np.exp(-(wx + w0[0][c]))
        denum= (1+partial)**2
        multi_factor= ((partial/denum))
        classwise=(multi_factor*(Y_truth[i,c] - Y_predicted[i,c]))
        a= a+classwise
        a=a*(-eta)
        
    return a







def gradient_W0(X, Y_truth, Y_predicted,W,w0):
    a=np.zeros([])
    b=np.array((3,2))
   
    c=0
    a=np.zeros([])
    for i in range(N):
        wx=(X[i][0]*W[0,c]+ X[i][1]*W[1,c])  
        partial=np.exp(-(wx + w0[0][c]))
        denum= (1+partial)**2
        multi_factor= (X[i]*(partial/denum))
        classwise=(multi_factor*(Y_truth[i,c] - Y_predicted[i,c]))
        a= a+classwise
        a=a*(-eta)
        
    return a

def gradient_W1(X, Y_truth, Y_predicted,W,w0):
    a=np.zeros([])
    b=np.array((3,2))
   
    c=1
    a=np.zeros([])
    for i in range(N):
        wx=(X[i][0]*W[0,c]+ X[i][1]*W[1,c])  
        partial=np.exp(-(wx + w0[0][c]))
        denum= (1+partial)**2
        multi_factor= (X[i]*(partial/denum))
        classwise=(multi_factor*(Y_truth[i,c] - Y_predicted[i,c]))
        a= a+classwise
        a=a*(-eta)
        
    return a

def gradient_W2(X, Y_truth, Y_predicted,W,w0):
    a=np.zeros([])
    b=np.array((3,2))
    
    c=2
    a=np.zeros([])
    for i in range(N):
        wx=(X[i][0]*W[0,c]+ X[i][1]*W[1,c])  
        partial=np.exp(-(wx + w0[0][c]))
        denum= (1+partial)**2
        multi_factor= (X[i]*(partial/denum))
        classwise=(multi_factor*(Y_truth[i,c] - Y_predicted[i,c]))
        a= a+classwise
        a=a*(-eta)
        
    return a
def gradient_W1c(X, Y_truth, Y_predicted,W,w0):
    a=np.zeros([])
    b=np.array((3,2))
   
    c=1
    a=np.zeros([])
    for i in range(N):
        wx=(X[i][0]*W[0,c]+ X[i][1]*W[1,c])  
        partial=np.exp(-(wx + w0[0][c]))
        denum= (1+partial)**2
        multi_factor= ((partial/denum))
        classwise=(multi_factor*(Y_truth[i,c] - Y_predicted[i,c]))
        a= a+classwise
        a=a*(-eta)
        
    return a


def gradient_W2c(X, Y_truth, Y_predicted,W,w0):
    a=np.zeros([])
    b=np.array((3,2))
   
    c=2
    a=np.zeros([])
    for i in range(N):
        wx=(X[i][0]*W[0,c]+ X[i][1]*W[1,c])  
        partial=np.exp(-(wx + w0[0][c]))
        denum= (1+partial)**2
        multi_factor= ((partial/denum))
        classwise=(multi_factor*(Y_truth[i,c] - Y_predicted[i,c]))
        a= a+classwise
        a=a*(-eta)
        
    return a

def grad(a,b,c):
    return(np.array[ [a] ,[b],[c] ])


# In[229]:


# learn W and w0 using gradient descent

iteration = 1
objective_values = []
while 1:
    Y_predicted = sigmoid(X, W, w0)
     
    objective_values = np.append(objective_values, np.sum( 0.5*(Y_truth -(Y_predicted))**2 ))
    W_old = W
    w0_old = w0
    # list_matrixc=[]
    #list_matrixc.append(gradient_W0c(X, Y_truth, Y_predicted,W,w0))
    # list_matrixc.append(gradient_W1c(X, Y_truth, Y_predicted,W,w0))
    #list_matrixc.append(gradient_W2c(X, Y_truth, Y_predicted,W,w0))
    #list_matrixc=np.array(list_matrixc)
  

    #list_matrix=[]
    # list_matrix.append(gradient_W0(X, Y_truth, Y_predicted,W,w0))
    #list_matrix.append(gradient_W1(X, Y_truth, Y_predicted,W,w0))
    #list_matrix.append(gradient_W2(X, Y_truth, Y_predicted,W,w0))
    #list_matrix=np.transpose(np.array(list_matrix))
  
    
  

    W = W - eta * gradient_w(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)
  #  W = W + list_matrix
   # w0 = w0 + list_matrixc

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print(W)
print(w0)


# In[230]:


# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[231]:


# calculate confusion matrix
y_predicted = np.argmax(Y_predicted, axis = 1) +1
print(y_predicted)
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[232]:


# evaluate discriminant function on a grid
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# In[ ]:





# In[ ]:




