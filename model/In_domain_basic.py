#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Five basic models
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.metrics import (calinski_harabaz_score,make_scorer, accuracy_score, precision_score, recall_score, matthews_corrcoef,roc_curve, auc)
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings,re
np.random.seed(0)
warnings.filterwarnings("ignore")


# In[2]:


def extract_data(tr_csv,te_csv,des_list=None):
    tr_initial = tr_csv
    te_initial = te_csv
    std_list = tr_initial.std()!=0
    tr = tr_initial.loc[:,std_list.index]
    te = te_initial.loc[:,std_list.index]
    if des_list == None:
        tr_x = tr.iloc[:,:-2]
        te_x = te.iloc[:,:-2]
        tr_cv = tr.iloc[:,:-1]
    else:
        tr_x = tr.loc[:,des_list]
        te_x = te.loc[:,des_list]
        
    tr_y = tr.iloc[:,-1]
    te_y = te.iloc[:,-1]
    return tr_x, tr_y.values, te_x, te_y.values, tr_cv
def data_scale(tr_x,te_x):
    if (tr_x.describe().loc['min',:].min() == 0 and
        tr_x.describe().loc['max',:].max() == 1):
        tr_scaled_x = tr_x
        te_scaled_x = te_x
    else:
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        tr_scaled_x = scaler.fit_transform(tr_x)
        te_scaled_x = scaler.transform(te_x)
    return tr_scaled_x,te_scaled_x
def show_metrics(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(int(y_true.shape[0])):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    se  = float(tp) / ( float(tp)+float(fn) )
    sp  = float(tn) / ( float(tn)+float(fp) )
    accuracy   = accuracy_score(y_true, y_pred)
    MCC        = matthews_corrcoef(y_true, y_pred)
    precision  = precision_score(y_true, y_pred)
    recall     = recall_score(y_true, y_pred)
    metrics=[tp,tn,fp,fn,se,sp,accuracy,precision,MCC,recall]
    return metrics
def forecast(clf,tr_scaled_x,te_scaled_x):
    tr_pre_y = clf.predict(tr_scaled_x)
    te_pre_y = clf.predict(te_scaled_x)
    tr_y_proba = clf.predict_proba(tr_scaled_x)[:,-1]
    te_y_proba = clf.predict_proba(te_scaled_x)[:,-1]
    return tr_pre_y,te_pre_y,tr_y_proba,te_y_proba
def evaluation(tr_pre_y,te_pre_y,tr_y,te_y):
    metrics_tr = show_metrics(tr_y, tr_pre_y)
    metrics_te = show_metrics(te_y, te_pre_y)
      
    result_info = {'tr_tp':metrics_tr[0],'tr_tn':metrics_tr[1],
                   'tr_fp':metrics_tr[2],'tr_fn':metrics_tr[3],
                  'tr_sensitivity':metrics_tr[4],'tr_specificity':metrics_tr[5],
                   'tr_accuracy':metrics_tr[6],'tr_MCC':metrics_tr[8],
                   'te_tp':metrics_te[0],'te_tn':metrics_te[1],
                   'te_fp':metrics_te[2],'te_fn':metrics_te[3],
                   'te_sensitivity':metrics_te[4],'te_specificity':metrics_te[5],
                   'te_accuracy':metrics_te[6],'te_MCC':metrics_te[8]}
    return result_info


# In[3]:


###Define the model optimization function
def KNN_gridsearch(tr_scaled_x,tr_y):
    warnings.filterwarnings("ignore")
    grid_dict = {'n_neighbors':range(1,50), 'weights':['distance', 'uniform'],"algorithm":['auto','ball_tree', 'kd_tree', 'brute']}

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)
    grid_estimator = KNeighborsClassifier(n_jobs=-1)
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer, 
                        cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_scaled_x, tr_y)
    return Grid.best_params_ ,Grid.best_estimator_
def DT_gridsearch(tr_scaled_x,tr_y):
    warnings.filterwarnings("ignore")
    criterion_list = ['gini','entropy']
    max_features_list = [None,'sqrt','log2']
    min_samples_split_list = range(2,11)

    grid_dict = {'criterion':criterion_list, 
                 'min_samples_split':min_samples_split_list,'max_features':max_features_list}

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=10, shuffle=True ,random_state=0)
    grid_estimator = DecisionTreeClassifier(random_state=0)
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer, 
                        cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_scaled_x, tr_y)
    return Grid.best_params_ ,Grid.best_estimator_
def SVC_gridsearch(tr_scaled_x,tr_y):
    warnings.filterwarnings("ignore")
    grid_list = []
    for i in range(-10,11):
        grid_list.append(2**i)
    grid_dict = {'C':grid_list, 'gamma':grid_list}
    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)    
    grid_estimator = SVC(probability=True)
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer, 
                        cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_scaled_x, tr_y)
    return Grid.best_params_ ,Grid.best_estimator_
def RF_gridsearch(tr_scaled_x,tr_y):
    warnings.filterwarnings("ignore")
    criterion_list = ['gini','entropy']
    max_features_list = [None,'sqrt','log2']
    min_samples_split_list = range(2,11)
 
    grid_dict = {'min_samples_split':min_samples_split_list,'criterion':criterion_list,
                 'max_features':max_features_list}
    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=10, shuffle=True ,random_state=0)
    grid_estimator = RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer, 
                        cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_scaled_x, tr_y)
    return Grid.best_params_ ,Grid.best_estimator_
def LG_gridsearch(tr_scaled_x,tr_y):
    warnings.filterwarnings("ignore")
    grid_dict = [{'penalty':["l1"], 'solver':['liblinear', 'saga']},
                 {'penalty':["l2"], 'solver':['newton-cg', 'lbfgs',  'sag','liblinear', 'saga']}]

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    grid_cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)
    grid_estimator = LogisticRegression(n_jobs=-1,random_state=0)
    Grid = GridSearchCV(grid_estimator, grid_dict, scoring=grid_scorer, 
                        cv=grid_cv, verbose=1,n_jobs=-1)
    Grid.fit(tr_scaled_x, tr_y)
    return Grid.best_params_ ,Grid.best_estimator_


# In[4]:


tr_y_all = np.empty([0,1], dtype=int)
te_y_all = np.empty([0,1], dtype=int)
tr_pre_y_all = np.empty([0,1], dtype=int)
te_pre_y_all = np.empty([0,1], dtype=int)
te_y_predproba_all = np.empty([0,1], dtype=int)
tr_y_predproba_all = np.empty([0,1], dtype=int)
every_output=pd.DataFrame(index=["method",'descriptors_num',
                             'tr_sensitivity','tr_specificity','tr_accuracy','tr_MCC','5_CV','10_CV','LOO',
                             'te_sensitivity','te_specificity','te_accuracy','te_MCC',
                             'tr_tp','tr_tn','tr_fp','tr_fn','te_tp','te_tn','te_fp','te_fn','best_params'])

for i in range(1,11):
    tr_csv = r'...\tr'+ str(i)+ r'_moe2d_pick_model.csv'
    te_csv = r'...\te'+ str(i)+ r'_moe2d_pick_model.csv'
    tr_x, tr_y, te_x, te_y = extract_data(tr_csv,te_csv) #descriptors_list
    tr_scaled_x, te_scaled_x = data_scale(tr_x, te_x)

    best_params, best_estimator = KNN_gridsearch(tr_scaled_x, tr_y)
    tr_pre_y,te_pre_y,tr_y_proba,te_y_proba = forecast(best_estimator,tr_scaled_x,te_scaled_x)
'''
###DT model
    best_params, best_estimator = DT_gridsearch(tr_scaled_x, tr_y,clus)
    tr_pre_y,te_pre_y,tr_y_proba,te_y_proba = forecast(best_estimator,tr_scaled_x,te_scaled_x)
###SVM model
    best_params, best_estimator = SVC_gridsearch(tr_scaled_x, tr_y,clus)
    tr_pre_y,te_pre_y,tr_y_proba,te_y_proba = forecast(best_estimator,tr_scaled_x,te_scaled_x)
###RF model
    best_params, best_estimator = RF_gridsearch(tr_scaled_x, tr_y,clus)
    tr_pre_y,te_pre_y,tr_y_proba,te_y_proba = forecast(best_estimator,tr_scaled_x,te_scaled_x)
###LC model
    best_params, best_estimator = LG_gridsearch(tr_scaled_x, tr_y,clus) 
    tr_pre_y,te_pre_y,tr_y_proba,te_y_proba = forecast(best_estimator,tr_scaled_x,te_scaled_x)
'''    
    every_result = evaluation(tr_pre_y,te_pre_y,tr_y,te_y)
    every_result['descriptors_num']=len(tr_x.columns)
    every_result['method']="KNN" ###Modified for different models
    every_result['best_params']=best_params
    every_out_csv_resulut = r"...\everyclus_moe2d_random_KNN.csv" #Modified for different models
    every_output[i-1] = pd.Series(every_result,dtype=np.object)
    
    tr_y_all = np.append(tr_y_all,tr_y)
    te_y_all = np.append(te_y_all,te_y)
    tr_pre_y_all = np.append(tr_pre_y_all,tr_pre_y)
    te_pre_y_all = np.append(te_pre_y_all,te_pre_y)
    te_y_predproba_all = np.append(te_y_predproba_all,te_y_proba) 
    tr_y_predproba_all = np.append(tr_y_predproba_all,tr_y_proba) 

every_output.T.to_csv(every_out_csv_resulut,index=False)
tr_pre_y_all=tr_pre_y_all.astype(np.int64)
te_pre_y_all=te_pre_y_all.astype(np.int64)
result = evaluation(tr_pre_y_all,te_pre_y_all,tr_y_all,te_y_all)
result['descriptors_num']=len(tr_x.columns)
result['method']="KNN"  #Modified for different models
output = pd.DataFrame(index=["method",'descriptors_num',
                             'tr_sensitivity','tr_specificity','tr_accuracy','tr_MCC',
                             'te_sensitivity','te_specificity','te_accuracy','te_MCC',
                             'tr_tp','tr_tn','tr_fp','tr_fn','te_tp','te_tn','te_fp','te_fn'])
output[0] = pd.Series(result,dtype=np.object)

out_csv_resulut = r"...\alldata_moe2d_random_KNN_ocv.csv" ###Modified for different models
output.T.to_csv(out_csv_resulut,index=False)

train_fpr,train_tpr,train_threshold = roc_curve(tr_y_all,tr_y_predproba_all,drop_intermediate=False)
train_roc_auc = auc(train_fpr,train_tpr)
test_fpr,test_tpr,stest_threshold = roc_curve(te_y_all,te_y_predproba_all,drop_intermediate=False)
test_roc_auc = auc(test_fpr,test_tpr)

plt.figure(figsize=(12,9))
plt.plot(train_fpr,train_tpr,'b',label='train_AUC = %0.5f'%train_roc_auc)
plt.plot(test_fpr,test_tpr,'g',label='test_AUC = %0.5f'%test_roc_auc)
plt.legend(loc='lower right',fontsize=15)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate',size=15)
plt.ylabel('True Positive Rate',size=15)
plt.title('KNN_MOE_KMEANS_ROC',size=25)###Modified for different models
plt.show()
#######The output path needs to be modified for different models
tr_y_predproba_all.tofile('KNN_tr_y_predproba_all.txt',sep=',',format='%10.5f')
te_y_predproba_all.tofile('KNN_te_y_predproba_all.txt',sep=',',format='%10.5f')
tr_y_all.tofile('KNN_tr_y_all.txt',sep=',',format='%d')
te_y_all.tofile('KNN_te_y_all.txt',sep=',',format='%d')
tr_pre_y_all.tofile('KNN_tr_pre_y_all.txt',sep=',',format='%d')
te_pre_y_all.tofile('KNN_te_pre_y_all.txt',sep=',',format='%d')

