#!/usr/bin/env python
# coding: utf-8

# 研究汇合后rocauc曲线绘制，以及保存proba等数据文件，为后续多模型roc绘制做准备

# In[1]:


import pandas as pd
import numpy as np
import copy
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,make_scorer,matthews_corrcoef,confusion_matrix,roc_curve,auc
from sklearn.model_selection import GridSearchCV,LeaveOneOut,cross_val_score


# In[2]:


def metric_report(test_y_pred_all,test_y_all,train_y_pred_all,train_y_all):
    test_accuracy = accuracy_score(test_y_pred_all,test_y_all)
    train_accuracy = accuracy_score(train_y_pred_all,train_y_all)
    test_conf_mat = confusion_matrix(test_y_all,test_y_pred_all)
    test_TP = test_conf_mat[1][1]
    test_TN = test_conf_mat[0][0]
    test_FP = test_conf_mat[0][1]
    test_FN = test_conf_mat[1][0]
    test_SE = test_TP/(test_TP+test_FN)
    test_SP = test_TN/(test_FP+test_TN)
    test_mcc = matthews_corrcoef(test_y_all,test_y_pred_all)
    train_conf_mat = confusion_matrix(train_y_all,train_y_pred_all)
    train_TP = train_conf_mat[1][1]
    train_TN = train_conf_mat[0][0]
    train_FP = train_conf_mat[0][1]
    train_FN = train_conf_mat[1][0]
    train_SE = train_TP/(train_TP+train_FN)
    train_SP = train_TN/(train_FP+train_TN)
    train_mcc = matthews_corrcoef(train_y_all,train_y_pred_all)
    print('Training set accuracy:%0.5f'%train_accuracy,'\n',
          'Training set sensitivity:%0.5f'%train_SE,'\n',
          'Training set specificity:%0.5f'%train_SP,'\n',
          'Training set MCC:%0.5f'%train_mcc,'\n',
          'Training set Confusion matrix:\n',train_conf_mat,'\n',
          'Test set accuracy:%0.5f'%test_accuracy,'\n',
          'Test set sensitivity:%0.5f'%test_SE,'\n',
          'Test set specificity:%0.5f'%test_SP,'\n',
          'Test set MCC:%0.5f'%test_mcc,'\n',
          'Test set Confusion matrix:\n',test_conf_mat,'\n')


# In[3]:


test_y_pred_all = np.empty([0,1], dtype=int)
train_y_pred_all = np.empty([0,1], dtype=int)
test_y_all = pd.Series()
train_y_all = pd.Series()
test_y_predproba_all = np.empty([0,2], dtype=int)
train_y_predproba_all = np.empty([0,2], dtype=int)


# In[5]:


for i in range(1,11):
    train_data_1 = pd.read_csv(r'...\tr'+ str(i)+ r'_moe2d_pick_model.csv')
    test_data_1 = pd.read_csv(r'...\te'+ str(i)+ r'_moe2d_pick_model.csv')
    train_data_copy = copy.deepcopy(train_data_1)
    test_data_copy = copy.deepcopy(test_data_1)
    train_data_random = train_data_copy.sample(frac=1,random_state=100)
    test_data_random = test_data_copy.sample(frac=1,random_state=100)
    train_x_index = train_data_random.columns[:-1]
    train_x0 = train_data_random[train_x_index]
    train_x = train_x0.drop(['Smiles'], axis = 1)
    train_y = train_data_random['Label']
    test_x_index = test_data_random.columns[:-1]
    test_x0 = test_data_random[test_x_index]
    test_x = test_x0.drop(['Smiles'], axis = 1)
    test_y = test_data_random['Label']
    candidate_learning_rate = [0.3]
    candidate_gamma = [0.11]
    candidate_max_depth = [5]
    candidate_n_estimators = [60]
    candidate_min_child_weight = [1]
    candidate_subsample = [0.6]
    candidate_colsample_bytree = [0.7]
    candidate_reg_alpha = [0]

    parameters = {'max_depth':candidate_max_depth,'min_child_weight':candidate_min_child_weight,'gamma':candidate_gamma,
              'subsample':candidate_subsample,'colsample_bytree':candidate_colsample_bytree,'n_estimators':candidate_n_estimators,
              'reg_alpha':candidate_reg_alpha,'learning_rate':candidate_learning_rate}
    model = xgb.XGBClassifier(eval_metric='error',seed=100,use_label_encoder=False,objective='binary:logistic')
    grid_scorer = make_scorer(matthews_corrcoef,greater_is_better=True)
    clf = GridSearchCV(model,parameters,scoring=grid_scorer,n_jobs=-1,cv=10)
    clf.fit(train_x,train_y)
    best_clf = clf.best_estimator_
    print(clf.best_params_)
    test_predict_results = best_clf.predict(test_x)
    train_predict_results = best_clf.predict(train_x)
    test_predict_pro = best_clf.predict_proba(test_x)
    train_predict_pro = best_clf.predict_proba(train_x)
    metric_report(test_predict_results,test_y,train_predict_results,train_y)
    test_y_pred_all = np.append(test_y_pred_all,test_predict_results)
    train_y_pred_all = np.append(train_y_pred_all,train_predict_results)
    test_y_all = test_y_all.append(test_y) 
    train_y_all = train_y_all.append(train_y) 
    test_y_predproba_all = np.append(test_y_predproba_all,test_predict_pro,axis=0)
    train_y_predproba_all = np.append(train_y_predproba_all,train_predict_pro,axis=0)
print("Evaluation of the outer ten-fold interactive test model：")
metric_report(test_y_pred_all,test_y_all,train_y_pred_all,train_y_all)
#The model parameters and evaluation of ten training sets test sets are output
train_y_all_save = np.array(train_y_all)
test_y_all_save = np.array(test_y_all)
train_y_all_save.tofile(r'...\MOE_train_y_all.txt',sep=',',format='%d')
test_y_all_save.tofile(r'...\MOE_test_y_all.txt',sep=',',format='%d')
test_y_pred_all.tofile(r'...\MOE_test_y_pred_all.txt',sep=',',format='%d')
train_y_pred_all.tofile(r'...\MOE_train_y_pred_all.txt',sep=',',format='%d')
test_y_predproba_all.tofile(r'...\MOE_test_y_predproba_all.txt',sep=',',format='%10.5f')
train_y_predproba_all.tofile(r'...\MOE_train_y_predproba_all.txt',sep=',',format='%10.5f')
test_y_prob_all = test_y_predproba_all[:,-1]
train_y_prob_all = train_y_predproba_all[:,-1]
test_y_prob_all.tofile(r'...\MOE_slice_test_y_prob_all.txt',sep=',',format='%10.5f')
train_y_prob_all.tofile(r'...\MOE_slice_train_y_prob_all.txt',sep=',',format='%10.5f')


# In[6]:


##Plot the ROC curve
train_fpr,train_tpr,train_threshold = roc_curve(train_y_all,train_y_prob_all,drop_intermediate=False)
train_roc_auc = auc(train_fpr,train_tpr)
test_fpr,test_tpr,stest_threshold = roc_curve(test_y_all,test_y_prob_all,drop_intermediate=False)
test_roc_auc = auc(test_fpr,test_tpr)
plt.figure(figsize=(12,9))
plt.plot(train_fpr,train_tpr,'b',label='train_AUC = %0.5f'%train_roc_auc)
plt.plot(test_fpr,test_tpr,'g',label='test_AUC = %0.5f'%test_roc_auc)
plt.legend(loc='lower right',fontsize=15)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate',size=15)
plt.ylabel('True Positive Rate',size=15)
plt.title('XGB_MOE_SPLIT_ROC',size=25)
plt.show()

