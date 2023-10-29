#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt


# In[38]:


def funca(x1, x2):
    return 2*x1*x1*x1 - 6*x2*x2 + 3*x1*x1*x2 


# In[39]:


x1, x2 = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
fig, ax = plt.subplots(1,1,figsize=(10,5))
cs = ax.contour(x1,x2,funca(x1,x2),levels=50)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()


# In[40]:


def funcb(x1,x2):
    return x1*x1 + (x1 + 1)*(x1*x1 + x2*x2)


# In[41]:


x1, x2 = np.meshgrid(np.linspace(-6,6,1000), np.linspace(-6,6,1000))
fig, ax = plt.subplots(1,1,figsize=(10,5))
cs = ax.contour(x1,x2,funcb(x1,x2),levels=150)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()


# In[42]:


def funcc(x1, x2):
    return np.log(1 + 0.5 * (x1*x1 + 3*x2*x2*x2))


# In[43]:


x1, x2 = np.meshgrid(np.linspace(-6,6,100), np.linspace(-0.85,6,100))
fig, ax = plt.subplots(1,1,figsize=(10,5))
cs = ax.contour(x1,x2,funcc(x1,x2),levels=50)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()


# In[44]:


def funcd(x1,x2):
    return (x1-2)*(x1-2) + x1*x2*x2 - 2 


# In[45]:


x1, x2 = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
fig, ax = plt.subplots(1,1,figsize=(10,5))
cs = ax.contour(x1,x2,funcd(x1,x2),levels=50)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()

