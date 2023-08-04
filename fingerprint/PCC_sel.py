#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PCC select for moe
import numpy as np
import pandas as pd


# In[2]:


input_file = 'alldata_moe2d.csv'
input_df = pd.read_csv(input_file)


# In[3]:


#PCC matrix is calculated, absolute value is taken,
# and sorted by absolute value of activity correlation coefficient
all_corr = input_df.corr().abs()
all_corr.sort_values(by=['Label'],axis=0,ascending=False,inplace=True)
all_corr.sort_values(by=['Label'],axis=1,ascending=False,inplace=True)


# In[4]:


#Find descriptors whose activity correlation is below the set value
activity_corr_low_id = np.where(all_corr.loc[:,'Label']<0.1)
activity_corr_low_list = list(all_corr.index[activity_corr_low_id])

#Filter out descriptors whose activity correlation is higher than the set value
a = all_corr.drop(labels=activity_corr_low_list+['Label'], axis=0)
activity_corr_high = a.drop(labels=activity_corr_low_list, axis=1)


# In[5]:


#Find out the pairs of descriptors whose ptwo correlation is greater than the set value, 
#and select the one with lower activity correlation
descriptor_corr_high_list = []
for i in range(len(activity_corr_high.index)):
    if activity_corr_high.iloc[i,:].name not in descriptor_corr_high_list:
        for j in range(i+2,len(activity_corr_high.columns)):
            if activity_corr_high.iloc[:,j].name not in descriptor_corr_high_list:
                if activity_corr_high.iloc[i,j] >=0.9:
                    descriptor_corr_high_list.append(activity_corr_high.iloc[:,j].name)


# In[6]:


#Merge the list of descriptors to be deleted to get the filtered list of descriptors
del_list = descriptor_corr_high_list+activity_corr_low_list+['Label']
all_names = all_corr.index

corr_result = all_names.drop(del_list)


# In[7]:


#The filtering results are stored in a CSV file
output_df = input_df.loc[:,list(corr_result)+['Label']]
output_df.to_csv(input_file[:-4]+'_corr_%d.csv'%(len(corr_result)),index=False)


# In[8]:


del_list


# In[9]:


corr_result


# In[10]:


#### corr_result stored in a CSV file
fobj = open(input_file[:-4]+'_corr.txt','w')
fobj.writelines(line + '\n' for line in corr_result)
fobj.close()

