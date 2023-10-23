#!/usr/bin/env python
# coding: utf-8

# In[142]:


import matplotlib.pyplot as plt
import numpy as np
import math 


# In[143]:


# read data into memory
data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header = 1)
x_test_points= np.zeros((150,1))
y_test_points= np.zeros((150,1))
every_x= np.zeros((272,1))



x_test_points_T= np.zeros((122,1))
y_test_points_T= np.zeros((122,1))
    
# get x and y values
for i in range(122):
    x_test_points_T[i] = data_set[150+i,0]
    every_x[i+150] = data_set[150+i,0]
    y_test_points_T[i] = data_set[150+i,1]
    



    
for i in range(150):
    x_test_points[i] = data_set[i,0]
    y_test_points[i] = data_set[i,1]
    every_x[i] = data_set[i,0]
    
    
x_train= np.transpose(x_test_points)[0]
y_train= np.transpose(y_test_points)[0]
x_test= np.transpose(x_test_points_T)[0]
y_test= np.transpose(y_test_points_T)[0]
_every= np.transpose(every_x)[0]



# get number of samples
N = x_test_points.shape[0]

#x_test = np.linspace(0,50, num = 8)


# In[144]:


bin_width= 0.37
left_borders= np.arange(1.5,5.2,bin_width)
right_borders= np.arange(1.5 +bin_width,5.2 +bin_width,bin_width)



# In[145]:



g_func = np.zeros((10,1))
g_func_fin = np.zeros(10)
    
# get x and y values




p_hat_denom = np.asarray([np.sum((left_borders[b] < np.transpose(x_test_points)[0]) & (np.transpose(x_test_points)[0] <= right_borders[b]))
                    for b in range(len(left_borders))])




for b in range(len(left_borders)):
    for i in range(150):
        if (left_borders[b] < (np.transpose((x_test_points))[0][i]))  & ((np.transpose((x_test_points))[0][i] <= right_borders[b])):
                                        g_func[b]=g_func[b] + np.transpose(y_test_points)[0][i]



    
use=np.transpose(g_func)[0].astype(int) /p_hat_denom


# In[146]:


plt.figure(figsize = (10, 6))
plt.plot( x_test_points, y_test_points, "b.", markersize = 10)
plt.plot( x_test_points_T, y_test_points_T, "r.", markersize = 10)
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min) ")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [use[b], use[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [use[b], use[b + 1]], "k-")    
plt.show()


error_func = np.zeros((122,1))

for i in range(122) :
    for b in range(len(left_borders)):
        if (left_borders[b] < (np.transpose((x_test_points_T))[0][i]))  & ((np.transpose((x_test_points_T))[0][i] <= right_borders[b])):
                                        error_func[i]=(np.transpose(y_test_points_T)[0][i]-use[b])**2
fin=math.sqrt(np.sum(error_func,axis=0)/122)
print("Regressogram => RMSE is  "+str(fin)+ " when h is 0.37")


# In[147]:


bin_width = 0.37

p_hat = np.asarray([np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in data_interval])
p_hat_mean = np.asarray([np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))*y_train ) for x in data_interval])

plt.figure(figsize = (10, 6))
plt.plot( x_test_points, y_test_points, "b.", markersize = 10)
plt.plot( x_test_points_T, y_test_points_T, "r.", markersize = 10)
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min) ")
plt.plot(data_interval, p_hat_mean/p_hat, "k-")
plt.show()
bin_width = 0.37

p_hat = np.asarray([np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in x_test])
p_hat_mean = np.asarray([np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))*y_train) for x in x_test])

fin= math.sqrt(np.sum(((y_test-(p_hat_mean/p_hat))**2))/122)
print("Running Mean Smoother => RMSE is "+str(fin)+ " when h is 0.37")


# In[148]:


bin_width = 0.37
p_hat = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) for x in data_interval])
p_hat_mean = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)*y_train) for x in data_interval])


plt.figure(figsize = (10, 6))
plt.plot( x_test_points, y_test_points, "b.", markersize = 10)
plt.plot( x_test_points_T, y_test_points_T, "r.", markersize = 10)
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min) ")
plt.plot(data_interval, p_hat_mean/p_hat, "k-")
plt.show()
bin_width = 0.37

p_hat = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) for x in x_test])
p_hat_mean = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)*y_train) for x in x_test])


fin= math.sqrt(np.sum(((y_test-(p_hat_mean/p_hat))**2))/122)
print("Kernel Smoother => RMSE is "+str(fin)+ " when h is 0.37")


# In[ ]:





# In[ ]:





# In[ ]:




