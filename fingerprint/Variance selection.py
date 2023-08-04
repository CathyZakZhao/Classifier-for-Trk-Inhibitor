#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##for MACCS and ECFP4 


# In[36]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


# In[37]:


#Extract descriptor information
path = r'all_data_finish_ECFP4.csv'
data=pd.read_csv(path)
desc_df = data.drop(['smiles','Label'],axis = 1)


# In[39]:


#Extract smiles and label
path = r'C:\Users\ZhaoXiaoman\Desktop\Computer aided drug screening\NTRK_FIRST CHOOSE\QSAR_paper one\experiment\1009\all_data_finish_ECFP4_2.csv'
data_1=pd.read_csv(path)
sti = data_1[['smiles','Label']]
sti.head()


# In[40]:


#Calculation selection
descs = desc_df.columns
X = desc_df.values
select = VarianceThreshold(threshold=0.01)
X_new = select.fit_transform(X)
desc_new_df = pd.DataFrame(X_new, columns=np.array(descs)[select.get_support()==True])


# In[42]:


#Merge information and output data
desc_pick_df = desc_new_df.append(sti)
desc_pick_df = pd.concat([sti,desc_new_df],axis=1)


# In[43]:


desc_pick_df.to_csv(path[:-4]+'pick.csv')

