#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs

class ECFP4:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def mol2fp(self, mol, radius = 2):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius,nBits=1024)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def compute_ECFP4(self, name):
        bit_headers = ['bit' + str(i) for i in range(1024)]
        arr = np.empty((0,1024), int).astype(int)
        for i in self.mols:
            fp = self.mol2fp(i)
            arr = np.vstack((arr, fp))
        df_ecfp4 = pd.DataFrame(np.asarray(arr).astype(int),columns=bit_headers)
        df_ecfp4.insert(loc=0, column='smiles', value=self.smiles)
        df_ecfp4.to_csv(name[:-4]+'_ECFP4.csv', index=False)

