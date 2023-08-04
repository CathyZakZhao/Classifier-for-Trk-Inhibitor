#!/usr/bin/env python
# coding: utf-8


from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles,SDMolSupplier,MACCSkeys,AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#Enter the data and compute similarity values
sdf_path = "all_data_finish.sdf"
ms = SDMolSupplier(sdf_path)
ms.SetProp('_Name',line.split()[1])
fps = [MACCSkeys.GenMACCSKeys(x) for x in ms]
hot_data = np.eye(len(fps))
for i in range(len(fps)):
    for j in range(i+1,len(fps)):
        hot_data[i,j] = DataStructs.FingerprintSimilarity(fps[i],fps[j])
        hot_data[j,i] = hot_data[i,j]
df_data= pd.DataFrame(hot_data)
df_data.to_csv('similarity.csv')

#Make a heat map of similarity distribution
plt.clf()
plt.figure(facecolor="w")
ax = plt.gca()
ax.xaxis.set_ticks_position("top") 
plt.imshow(hot_data,cmap='rainbow')  
plt.colorbar()
plt.savefig("MACCSout1.png", dpi=300, bbox_inches="tight")
plt.show()

#Make a histogram of similarity value distribution
hist = np.triu(hot_data,1).flatten()
hist = hist[hist>0]
plt.rcParams['font.family'] = ['Times New Roman']# 设置字体
plt.clf()
plt.figure(facecolor="w")
plt.hist(hist,bins=50,range=(0,1),weights= [1./ len(hist)]*len(hist),facecolor="k",alpha=0.6)
def to_percent(y, position):
    return '%1.1f'%(100*y) + '%'
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.ylabel("Frequncy")
plt.xlabel("MACCS_Tc")
plt.grid(True)
plt.savefig("MACCSout2.png", dpi=300, bbox_inches="tight")
plt.show()

