#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef,accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
import copy
import numpy as np
import os


# In[2]:


###Data preprocessing
def prepare(data,activity_name):
    data_copy = copy.deepcopy(data)
    x_df = data_copy.iloc[:,1:-2]
    y_df = data_copy[str(activity_name)]
    x_arr = x_df.values
    y_arr = y_df.values
    y_arr = encode(y_arr)
    x = torch.tensor(x_arr).float()
    y = torch.tensor(y_arr).float()
    return x,y

def encode(arr):
    arr = np.array(arr).reshape(len(arr),-1)
    enc = OneHotEncoder()
    enc.fit(arr)
    target = enc.transform(arr).toarray()
    return target

def decode(arr):
    l = []
    for i in range(len(arr)):
        if arr[i][0] > 0.5:
            l.append(1)
        elif arr[i][0] < 0.5:
            l.append(0)
        else:
            l.append('error')
    return l

def accuracy(y_pred,y_true):
    acc_amount = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            acc_amount += 1
        else:
            continue
    return acc_amount


# In[3]:


###Training network               
def fit(model,loss_func,optimizer,train_loader,test_loader,
        save_path,epochs,cuda=True,*device):
    x_epoch = []
    y_tr_acc = []
    y_te_acc = []
    te_acc = []
    bestnet_num = int()
    for epoch in range(epochs):
        acc_amount_tr = 0
        train_amount = 0
        acc_amount_te = 0
        test_amount = 0
        for i,data in enumerate(train_loader,0):
            model.zero_grad()
            inputs,labels = data
            if cuda == True:
                inputs = inputs.to(device[0])
                labels = labels.to(device[0])
            else:
                pass
            y_pred_tr = model(inputs)
            loss = loss_func(y_pred_tr,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cuda == True:
                y_pred_tr = y_pred_tr.to('cpu')
                labels = labels.to('cpu')
            else:
                pass
            y_pred_arr_tr = y_pred_tr.data.numpy().tolist()
            l_pred_tr = decode(y_pred_arr_tr)
            l_labels_tr = decode(labels)
            acc_batch_tr = accuracy(l_pred_tr,l_labels_tr)
            acc_amount_tr += acc_batch_tr
            train_amount += len(l_labels_tr)
        acc_tr = (acc_amount_tr/train_amount)*100
        print('***************************************\n',
            epoch,'train_acc:%.2f'%acc_tr,'%')
        with torch.no_grad():
            model.eval()
            for j,data in enumerate(test_loader,0):
                inputs,labels = data
                if cuda == True:
                    inputs = inputs.to(device[0])
                else:
                    pass
                y_pred_te = model(inputs)
                if cuda == True:
                    y_pred_te = y_pred_te.to('cpu')
                else:
                    pass
                y_pred_arr_te = y_pred_te.data.numpy().tolist()
                l_pred_te = decode(y_pred_arr_te)
                l_labels_te = decode(labels)
                acc_batch_te = accuracy(l_pred_te,l_labels_te)
                acc_amount_te += acc_batch_te
                test_amount += len(l_labels_te)
        acc_te = (acc_amount_te/test_amount)*100
        te_acc.append(acc_te)
        max_acc_te = max(te_acc)
        bestnet_num = te_acc.index(max_acc_te)
        print(epoch,'test_acc:%.2f'%acc_te,'%\n',
              'Maximum accuracy of test:%.2f'%max_acc_te,'% in epoch',bestnet_num)
        model_path = save_path+'\\'+str(epoch)+r".pth"
        torch.save(model,model_path)
        x_epoch.append(epoch)
        y_tr_acc.append(acc_tr)
        y_te_acc.append(acc_te)
        plt.figure(figsize=(8,4))
        plt.plot(x_epoch,y_tr_acc,label='train_acc',color='blue',lw=1)
        plt.plot(x_epoch,y_te_acc,label='test_acc',color='green',lw=1)
        plt.xlabel('epoch')
        plt.ylabel(r'acc/%')
        plt.title('Epoch-Accuracy_rate')
        plt.ylim(50,100)
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()
    return bestnet_num


# In[4]:


###Output the optimal model index      
def model_pred(best_model,loader,cuda=True,*device):
    with torch.no_grad():
        l_labels = []
        l_y_pred = []
        l_y_score = []
        for inputs,labels in loader:
            if cuda == True:
                inputs = inputs.to(device[0])
            else:
                pass
            y_pred = best_model(inputs)
            if cuda == True:
                y_pred = y_pred.to('cpu')
            else:
                pass
            y_pred = y_pred.numpy().tolist()
            for i in range(len(y_pred)):
                l_y_score.append(y_pred[i][0])
            l_batch_pred = decode(y_pred)
            l_y_pred += l_batch_pred
            l_batch_labels = decode(labels)
            l_labels += l_batch_labels
    return l_labels,l_y_pred,l_y_score


# In[5]:


def SE_and_SP(conf_mat):
        TP = conf_mat[1][1]
        TN = conf_mat[0][0]
        FP = conf_mat[0][1]
        FN = conf_mat[1][0]
        SE = TP/(TP+FN)
        SP = TN/(FP+TN)
        return SE,SP


# In[6]:


def report(y_true,y_pred,title):
    report = sklearn.metrics.classification_report(y_true,y_pred)
    print(title,'report:\n',report)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true,y_pred)
    print(title,'confusion_matrix:\n',confusion_matrix)
    mcc = matthews_corrcoef(y_true, y_pred)
    SE,SP = SE_and_SP(confusion_matrix)
    acc = accuracy_score(y_pred,y_true)*100
    print(str(title)+'_acc: %.2f'%acc,'%\n',
          str(title)+'_mcc: %.3f'%mcc,'\n',
          str(title)+'_SE: %.3f'%SE,'\n',
          str(title)+'_SP: %.3f'%SP)


# In[7]:


def get_wrong(model,X_tensor,y_tensor,cuda=True,*device):
    with torch.no_grad():
        if cuda == True:
            y_pred = model(X_tensor.to(device[0]))
        else:
            y_pred = model(X_tensor)
        y_pred = decode(y_pred)
        y_true = decode(y_tensor)
        dif = []
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                continue
            else:
                dif.append(i)
    return dif


# In[8]:


##Defined network
class MyNet(nn.Module):
    
    def __init__(self,inputs):
        super(MyNet,self).__init__()
        self.fc_1 = nn.Linear(inputs,120)
        self.bn_1 = nn.BatchNorm1d(120)
        self.fc_2 = nn.Linear(120,160)
        self.bn_2 = nn.BatchNorm1d(160)
        self.fc_3 = nn.Linear(160,120)
        self.bn_3 = nn.BatchNorm1d(120)
        self.fc_4 = nn.Linear(120,80)
        self.bn_4 = nn.BatchNorm1d(80)
        self.fc_5 = nn.Linear(80,20)
        self.bn_5 = nn.BatchNorm1d(20)
        self.fc_6 = nn.Linear(20,2)
        
    def forward(self,input_data):
        x = F.relu(self.bn_1(self.fc_1(input_data)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = F.relu(self.bn_3(self.fc_3(x)))
        x = F.relu(self.bn_4(self.fc_4(x)))
        x = F.relu(self.bn_5(self.fc_5(x)))
        x = torch.sigmoid(self.fc_6(x))
        return x


# In[9]:


if __name__ == '__main__':
##GPU acceleration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#-----------------------------------------------------------------------------#
 
    test_y_pred_all = []
    train_y_pred_all = []
    test_y_all = []
    train_y_all = []
    test_y_score_all = []
    train_y_score_all = []
#-----------------------------------------------------------------------------#    
###The following two data reads are selective for different types of models

####data for cross-domain model
    all_csv = pd.read_csv('alldata_moe2d_pick_kmeans.csv')
    for i in range(10):
        tr_csv_initial = all_csv[all_csv['km_clus']!=i]
        te_csv_initial = all_csv[all_csv['km_clus']==i]
        X_train,y_train = prepare(tr_csv_initial,'Label')
        X_test,y_test = prepare(te_csv_initial,'Label')

####data for in-domain model
#for i in range(1,11):
#        X_train,y_train = prepare(r'...\tr'+ str(i)+ r'_moe2d_pick_model.csv','Label')
 #       X_test,y_test = prepare(r'...\te'+ str(i)+ r'_moe2d_pick_model.csv','Label')

#-----------------------------------------------------------------------------#
##Construct the data set and the loader       
        train_dataset = TensorDataset(X_train,y_train)
        test_dataset = TensorDataset(X_test,y_test)
        train_loader = DataLoader(dataset=train_dataset,batch_size=25,shuffle=True,num_workers=0)#shuffle = True可能报错,也可能因为num_works != 0报错
        test_loader = DataLoader(dataset=test_dataset,batch_size=25,shuffle=True,num_workers=0)
###instantiation       
        net = MyNet(55).to(device)
        optimizer = torch.optim.Adam(net.parameters(),lr=0.001,weight_decay=0.01)
        loss_func = torch.nn.BCELoss().to(device)
###Create folder
        os.mkdir(r'...\DNN'+ str(i))
#Train and save the model    
        bestnet_num = fit(net,loss_func,optimizer,train_loader,test_loader,r'...\DNN'+ str(i),100,True,device)
        best_net = torch.load(r'...\DNN'+ str(i)+r"\\" +str(bestnet_num)+ r'.pth') 
        print(best_net)
#The optimal model is used for prediction
        train_true,train_pred,train_score = model_pred(best_net,train_loader,True,device)
        test_true,test_pred,test_score = model_pred(best_net,test_loader,True,device)
        report(train_true,train_pred,'train')
        report(test_true,test_pred,'test')
        test_y_pred_all += test_pred
        train_y_pred_all += train_pred
        test_y_all += test_true 
        train_y_all += train_true
        test_y_score_all += test_score
        train_y_score_all += train_score
    report(train_y_all,train_y_pred_all,'train_all')
    report(test_y_all,test_y_pred_all,'test_all')
#plot
    train_fpr,train_tpr,train_thresholds = sklearn.metrics.roc_curve(train_y_all,train_y_score_all)
    train_auc = sklearn.metrics.auc(train_fpr, train_tpr)
    test_fpr,test_tpr,test_thresholds = sklearn.metrics.roc_curve(test_y_all,test_y_score_all)
    test_auc = sklearn.metrics.auc(test_fpr,test_tpr)
    plt.figure(figsize=(12,9))
    plt.plot(train_fpr,train_tpr,'b',label='train_AUC = %0.4f'%train_auc)
    plt.plot(test_fpr,test_tpr,'g',label='test_AUC = %0.4f'%test_auc)
    plt.legend(loc='lower right',fontsize=15)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False Positive Rate',size=15)
    plt.ylabel('True Positive Rate',size=15)
    plt.title('DNN_KMENANS_MOE_ROC',size=25)
    plt.show()
#Output to txt files    
    test_y_pred_all = np.array(test_y_pred_all)
    test_y_pred_all.tofile(r'...\test_y_pred_all.txt',sep=',',format='%d')
    train_y_pred_all = np.array(train_y_pred_all)
    train_y_pred_all.tofile(r'...\train_y_pred_all.txt',sep=',',format='%d')
    test_y_all = np.array(test_y_all)
    test_y_all.tofile(r'...\test_y_all.txt',sep=',',format='%d')
    train_y_all = np.array(train_y_all)
    train_y_all.tofile(r'...\train_y_all.txt',sep=',',format='%d')
    test_y_score_all = np.array(test_y_score_all)
    test_y_score_all.tofile(r'...\test_y_score_all.txt',sep=',',format='%10.5f')
    train_y_score_all = np.array(train_y_score_all)
    train_y_score_all.tofile(r'...\train_y_score_all.txt',sep=',',format='%10.5f')


# In[ ]:




