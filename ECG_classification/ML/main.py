# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 03:24:13 2020

@author: SWH
"""
#%%
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import entropy
import numpy as np

SEED = 777
np.random.seed(SEED)

EPOCH = 10

fold_num = 5
INDEX = 600
data = data.sort_values(['23'], ascending=[True])
data = data.reset_index(drop=True) # 각 피쳐할때는 .values 지우기
input_x = data[['8', '12', '14']] #[0,1],2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 or
input_y = data['23'] # '23' or

norbeat = input_x[:11020]
norbeat_l = input_y[:11020]
arrthybeat = input_x[11028:108328]#
arrthybeat_l = input_y[11028:108328]

x = np.concatenate([norbeat, arrthybeat], axis=0) #.to_numpy()
y = np.concatenate([norbeat_l, arrthybeat_l], axis=0)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=100)
acc = cross_val_score(clf, x, y, cv=kf, n_jobs=4, verbose=1)
pre = cross_val_score(clf, x, y, cv=kf, n_jobs=4, verbose=1, scoring='precision')
recall = cross_val_score(clf, x, y, cv=kf, n_jobs=4, verbose=1, scoring='recall')

acc_mean = np.mean(acc)
pre_mean = np.mean(pre)
recall_mean = np.mean(recall)

print('acc mean: {}'.format(acc_mean))
print('pre mean: {}'.format(pre_mean))
print('recall mean: {}'.format(recall_mean))
