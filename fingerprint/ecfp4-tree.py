#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install pydotplus')


# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef
from rdkit.Chem import AllChem
from rdkit import Chem
import pydotplus
import copy


# In[2]:


if __name__ == '__main__':
    all_data_path = 'all_data_finish_ECFP4_pick.csv'
    #这里应该输入未筛选的描述符
    som_train_data_path = 'tr1_ecfp4.csv'
    all_data = copy.deepcopy(pd.read_csv(all_data_path))
    train_data = copy.deepcopy(pd.read_csv(som_train_data_path))
    train_smiles = train_data['smiles']
    train_y = train_data['Label']
    all_smiles = all_data['smiles']
    all_x = all_data.iloc[:, 2:]
    all_y = all_data['Label']
    index_list0 = all_data.columns.values
    index_list = index_list0[2:]
    train_x = train_data.iloc[:, 2:]


# In[3]:


index_list


# In[4]:


len(index_list)


# In[8]:


candidate_criterion = ['entropy', 'gini']
candidate_max_features = [None, 'sqrt', 'log2']
candidate_max_depth = [4]  #可以适当增加层数
candidate_max_leaf_nodes = range(10, 30)
params = {'criterion': candidate_criterion, 'max_features': candidate_max_features,
     'max_depth': candidate_max_depth, 'max_leaf_nodes': candidate_max_leaf_nodes}


# In[9]:


classifier = DecisionTreeClassifier(random_state=100)
grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
clf = GridSearchCV(classifier, params, scoring=grid_scorer, n_jobs=-1, cv=10)
clf.fit(train_x, train_y)
best_clf = clf.best_estimator_
best_parameter = clf.best_params_
print(best_parameter)
best_clf.fit(all_x, all_y)
pred = best_clf.predict(all_x)
acc = accuracy_score(all_y, pred)
print(acc * 100)


# In[10]:


dot_data= export_graphviz(best_clf, out_file=None, feature_names =index_list, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("ecfp4pick_tree_depth4.pdf")


# In[11]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

