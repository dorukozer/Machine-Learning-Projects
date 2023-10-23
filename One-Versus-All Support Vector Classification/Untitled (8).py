#!/usr/bin/env python
# coding: utf-8

# In[202]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt


# In[203]:


data_set = np.genfromtxt("hw06_images.csv", delimiter = ",")
y = np.genfromtxt("hw06_labels.csv", delimiter = ",")


# In[204]:


# get X and y values
X_train = data_set[0:1000,:]
X_test = data_set[1000:5000,:]
y_train = y[0:1000].astype(int)
y_test= y[1000:5000].astype(int)



# In[205]:


N_train = len(y_train)
D_train = X_train.shape[1]


# In[206]:


def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)

def maximizer(y_train1, K_train,s,C):
    yyK = np.matmul(y_train1[:,None], y_train1[None,:]) * K_train
    # set learning parameters
    epsilon = 1e-3
    N_train=1000
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((1000, 1)))
    G = cvx.matrix(np.vstack((-np.eye(1000), np.eye(1000))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((1000, 1)))))
    A = cvx.matrix(1.0 * y_train1[None,:])
    b = cvx.matrix(0.0)  
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train1[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    # find bias parameter
    return np.array([alpha,w0],dtype=object)


   


# In[207]:



# calculate Gaussian kernel
s=10
C=10
yclass = np.zeros((1000,5))
                   

for i in range(0,len(y_train)):
    yclass[i][y_train[i]-1]=1
yclass=2*np.array(yclass)-1


SVM1= np.transpose(yclass)[0]
SVM2= np.transpose(yclass)[1]
SVM3= np.transpose(yclass)[2]
SVM4= np.transpose(yclass)[3]
SVM5= np.transpose(yclass)[4]

K_train = gaussian_kernel(X_train, X_train, 10)

alpha1,w01= maximizer(SVM1, K_train,10,10)
alpha2,w02= maximizer(SVM2, K_train,10,10)
alpha3,w03= maximizer(SVM3, K_train,10,10)
alpha4,w04= maximizer(SVM4, K_train,10,10)
alpha5,w05= maximizer(SVM5, K_train,10,10)










# In[208]:


y_Result=np.zeros(1000)
f_predicted1 = np.matmul(K_train, SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_train,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_train, SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_train,  SVM4[:,None] *  alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_train,  SVM5[:,None] * alpha5[:,None]) + w05
a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1
# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


accuracy10=np.zeros(5)
for i in range(5):
     accuracy10[2] += np.array([confusion_matrix])[0][i][i]


accuracy10[2]=accuracy10[2]/1000




# In[209]:


K_test = gaussian_kernel(X_test,X_train, s)


f_predicted1 = np.matmul(K_test,  SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_test,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_test,  SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_test,  SVM4[:,None] * alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_test,  SVM5[:,None] * alpha5[:,None]) + w05

a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1


# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)

accuracy10t=np.zeros(5)
for i in range(5):
     accuracy10t[2] += np.array([confusion_matrix])[0][i][i]


accuracy10t[2]=accuracy10t[2]/4000


# In[210]:


alpha1,w01= maximizer(SVM1, K_train,10,0.1)
alpha2,w02= maximizer(SVM2, K_train,10,0.1)
alpha3,w03= maximizer(SVM3, K_train,10,0.1)
alpha4,w04= maximizer(SVM4, K_train,10,0.1)
alpha5,w05= maximizer(SVM5, K_train,10,0.1)





y_Result=np.zeros(1000)
f_predicted1 = np.matmul(K_train, SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_train,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_train, SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_train,  SVM4[:,None] *  alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_train,  SVM5[:,None] * alpha5[:,None]) + w05
a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1
# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


for i in range(5):
     accuracy10[0] += np.array([confusion_matrix])[0][i][i]


accuracy10[0]=accuracy10[0]/1000

K_test = gaussian_kernel(X_test,X_train, s)


f_predicted1 = np.matmul(K_test,  SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_test,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_test,  SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_test,  SVM4[:,None] * alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_test,  SVM5[:,None] * alpha5[:,None]) + w05

a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1


# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])



for i in range(5):
     accuracy10t[0] += np.array([confusion_matrix])[0][i][i]


accuracy10t[0]=accuracy10t[0]/4000


# In[211]:


alpha1,w01= maximizer(SVM1, K_train,10,1)
alpha2,w02= maximizer(SVM2, K_train,10,1)
alpha3,w03= maximizer(SVM3, K_train,10,1)
alpha4,w04= maximizer(SVM4, K_train,10,1)
alpha5,w05= maximizer(SVM5, K_train,10,1)






y_Result=np.zeros(1000)
f_predicted1 = np.matmul(K_train, SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_train,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_train, SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_train,  SVM4[:,None] *  alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_train,  SVM5[:,None] * alpha5[:,None]) + w05
a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1
# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)

accuracy1=0
for i in range(5):
     accuracy10[1] += np.array([confusion_matrix])[0][i][i]


accuracy10[1]=accuracy10[1]/1000



K_test = gaussian_kernel(X_test,X_train, s)


f_predicted1 = np.matmul(K_test,  SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_test,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_test,  SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_test,  SVM4[:,None] * alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_test,  SVM5[:,None] * alpha5[:,None]) + w05

a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1


# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


for i in range(5):
     accuracy10t[1] += np.array([confusion_matrix])[0][i][i]


accuracy10t[1]=accuracy10t[1]/4000


# In[212]:


alpha1,w01= maximizer(SVM1, K_train,10,100)
alpha2,w02= maximizer(SVM2, K_train,10,100)
alpha3,w03= maximizer(SVM3, K_train,10,100)
alpha4,w04= maximizer(SVM4, K_train,10,100)
alpha5,w05= maximizer(SVM5, K_train,10,100)






y_Result=np.zeros(1000)
f_predicted1 = np.matmul(K_train, SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_train,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_train, SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_train,  SVM4[:,None] *  alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_train,  SVM5[:,None] * alpha5[:,None]) + w05
a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1
# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


for i in range(5):
     accuracy10[3] += np.array([confusion_matrix])[0][i][i]


accuracy10[3]=accuracy10[3]/1000


K_test = gaussian_kernel(X_test,X_train, s)


f_predicted1 = np.matmul(K_test,  SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_test,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_test,  SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_test,  SVM4[:,None] * alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_test,  SVM5[:,None] * alpha5[:,None]) + w05

a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1


# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


for i in range(5):
     accuracy10t[3] += np.array([confusion_matrix])[0][i][i]


accuracy10t[3]=accuracy10t[3]/4000


# In[213]:


alpha1,w01= maximizer(SVM1, K_train,10,1000)
alpha2,w02= maximizer(SVM2, K_train,10,1000)
alpha3,w03= maximizer(SVM3, K_train,10,1000)
alpha4,w04= maximizer(SVM4, K_train,10,1000)
alpha5,w05= maximizer(SVM5, K_train,10,1000)
#w01= maximizer(SVM1, K_train,10,1000)[1] 
#w02= maximizer(SVM2, K_train,10,1000)[1]
#w03= maximizer(SVM3, K_train,10,1000)[1]
#w04= maximizer(SVM4, K_train,10,1000)[1]
#w05= maximizer(SVM5, K_train,10,1000)[1]



y_Result=np.zeros(1000)
f_predicted1 = np.matmul(K_train, SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_train,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_train, SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_train,  SVM4[:,None] *  alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_train,  SVM5[:,None] * alpha5[:,None]) + w05
a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1
# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)

for i in range(5):
     accuracy10[4] += np.array([confusion_matrix])[0][i][i]


accuracy10[4]=accuracy10[4]/1000



K_test = gaussian_kernel(X_test,X_train, s)


f_predicted1 = np.matmul(K_test,  SVM1[:,None] * alpha1[:,None]) + w01
f_predicted2 = np.matmul(K_test,  SVM2[:,None] * alpha2[:,None]) + w02
f_predicted3 = np.matmul(K_test,  SVM3[:,None] * alpha3[:,None]) + w03
f_predicted4 = np.matmul(K_test,  SVM4[:,None] * alpha4[:,None]) + w04
f_predicted5 = np.matmul(K_test,  SVM5[:,None] * alpha5[:,None]) + w05

a= np.hstack((f_predicted1 , f_predicted2,  f_predicted3 , f_predicted4,  f_predicted5))
result  = np.argmax(a, axis=1)+1


# calculate confusion matrix
confusion_matrix = pd.crosstab(np.reshape(result, 4000), y_test, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)


for i in range(5):
     accuracy10t[4] += np.array([confusion_matrix])[0][i][i]


accuracy10t[4]=accuracy10t[4]/4000


# In[214]:


rangex=np.array([0.1,1,10,100,1000])
# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(rangex, accuracy10, "r-")
plt.plot(rangex, accuracy10, ".r", markersize = 10)
plt.plot(rangex, accuracy10t, ".b", markersize = 10)
plt.plot(rangex, accuracy10t, "b-")
plt.xscale("log")
plt.xlabel("Regularazition paramter C")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




