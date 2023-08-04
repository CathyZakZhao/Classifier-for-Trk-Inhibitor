#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,SpectralClustering
from sklearn import metrics
import numpy.random as nr
import math
from scipy.spatial.distance import cdist


# In[3]:


###“Activity”
all_init_df = pd.read_csv("all_data_finish_MACCS.csv")
all_maccs = all_init_df.iloc[:, 1:-1].values
weak_samples = all_maccs[np.where(all_init_df.iloc[:,-1] == 0)]
high_samples = all_maccs[np.where(all_init_df.iloc[:,-1] == 1)]


# In[4]:


#tsne
tsne=TSNE(random_state=0,perplexity=30,init='pca')
tsne.fit(all_maccs)
X = tsne.embedding_


# In[5]:


###k
Cluster = KMeans(n_clusters=10,random_state=7, max_iter=200,n_jobs=-1).fit(X)
y_pred = Cluster.predict(X)


# In[11]:


y_pred.tofile('kmeanscluster_y_pred.txt',sep=',',format='%d')


# In[38]:


color = ['#5B58A7','#3489BB','#60C4A6','#9FD3A3','#DCE899','#FDDC89','#FAA966','#F3754E','#D84858','#9A154C']
marker = ['o','+','1','^','*','p','d','3','1','2']
plt.clf()
points = Cluster.cluster_centers_
plt.scatter(points[:, 0], points[:, 1], s=50, c='r', marker='x',zorder=5,label='center_compounds')
for i in range(10):
    plt.plot(X[:,0][np.where(y_pred==i)], X[:,1][np.where(y_pred==i)], 
                marker=marker[i],c=color[i],linestyle='',label='subset'+'%d'%(i))

num1 = 1.05
num2 = 0
num3 = 3
num4 = 0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
#plt.gca().add_artist(le1)
#plt.legend(loc='right',ncol=2)
plt.xlabel('TSNE1',size=12)
plt.ylabel('TSNE2',size=12)
#plt.title('KMEANS_CLUSTER',size=20)
plt.savefig('kmeanscluster.png',dpi=600, bbox_inches="tight")
plt.show()


# In[7]:


Cluster.cluster_centers_


# In[ ]:


#for k values
score = 0 
score_id = 0
best_y_pred = np.array([])
for i in range(2,10):
    y_pred = KMeans(n_clusters=i,random_state=0).fit_predict(X)
    if score < metrics.calinski_harabaz_score(X, y_pred):
        score = metrics.calinski_harabaz_score(X, y_pred)
        score_id = i
        best_y_pred = y_pred
print(score)
print(score_id)


# In[ ]:


#for k values(optional)
K = range(1,20)
meandistortions = []
for k in K:
    
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])  
    
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('Ave Distor')
plt.show()


# In[ ]:




