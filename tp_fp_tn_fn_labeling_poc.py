
from scipy import interp
import numpy as np
import pandas as pd
from itertools import cycle
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc, recall_score, confusion_matrix, precision_score, accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == '__main__':

#_________________________________________
    # definding FP and TP values
    #__________________________________________

    # quick check of TP and FP count
    tn, fp, fn, tp = confusion_matrix(y_true[0:, 0], y_pred1[0:, 0]).ravel()
    print(f'FP{fp}\nTP {tp}\nTN {tn}\nFN {fn}')

    #creating a base df
    x_test_all = pd.DataFrame(x_test).copy()

    #adding predict proba as feature into x_test
    x_test_all[['pred_prob0','pred_prob1']] = pd.DataFrame(y_score).copy()

    #adding to base df  2 columns with ytrue==1 and ytrue==0
    x_test_all[['y_true0','y_true1']] = pd.DataFrame(y_true).copy()

    #the same as for ytrue, but 2 columns into base df for predicted values
    x_test_all[['ypred0','ypred1']] = pd.DataFrame(y_pred1).copy()

    #adding columns with FP and TP; nan if row doesn't contains value
    x_test_all['tp'] = x_test_all['y_true0'][(x_test_all['y_true0'] == 1 ) &
                                             (x_test_all['y_true0'] == x_test_all['ypred0'].apply(int))]

    x_test_all['fp'] = x_test_all['y_true0'][(x_test_all['y_true0'] == 0 ) &
                                             (x_test_all['y_true0'] != x_test_all['ypred0'].apply(int))]

    #adding columns with  TN; nan if row doesn't contains value, 0 - if contains
    x_test_all['tn'] = x_test_all['y_true0'][(x_test_all['y_true0'] == 0 ) &
                                             (x_test_all['y_true0'] == x_test_all['ypred0'].apply(int))]

    #reduce the rows: leave only with FP and FN
    x_set_new_TP_FP = x_test_all[x_test_all['tp'].notna() | x_test_all['fp'].notna()]

    #reduce the rows: leave only with TP and TN
    x_set_new_TP_TN = x_test_all[x_test_all['tp'].notna() | x_test_all['tn'].notna()]

    #labeling TP as 1, TN as 2
    x_set_new_TP_TN.tp.fillna(2,inplace=True)

    # create variables to return for TP_FP
    x_test_new_TP_FP = x_set_new_TP_FP.iloc[:,0:-7].copy().to_numpy()

    # create variables to return for TP_TN
    x_test_new_TP_TN = x_set_new_TP_TN.iloc[:,0:-7].copy()
    x_test_new_TP_TN['label'] = x_set_new_TP_TN.iloc[:,-3].copy()
    x_test_new_TP_TN.to_numpy()

    #redusing rows with only FP for KNN
    x_set_only_fp = x_test_all[x_test_all['fp'] == 0].iloc[:,0:-9].copy().to_numpy()

    #labeling TP as 1, FP as 0
    x_set_new_TP_FP.fp.fillna(1, inplace=True)

    #making new y 1d array
    y_test_new = 1 - x_set_new_TP_FP.fp.copy().to_numpy()