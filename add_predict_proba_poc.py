import pickle
import sys
from datetime import datetime as dt
import numpy as np
import os
# import ray
import pandas as pd


def run_only_final(model_name, model, X_test, Y_test , GAP_TIME, flag=False):
    """
   The function gets: model_name str, model class, X_test (or X_val)  and Y_set ndarrays, gap_time int, flag bool.
   After building the model on the train set.
   If flag=True --> dump with name 'test', otherwise 'val'
   Function saves rusults into csv
   returns:  dataframe predict_proba + 3 filepath pkl for real labels, predicted labels and pred probabilities
   """
    mksureDir('./Aucroc')
    y_pred_test = pd.DataFrame(model.predict_proba(X_test))
    y_predict_test = pd.DataFrame(model.predict(X_test))
    Y_test = pd.DataFrame(Y_test)

    if not flag:
        # SAVE results for test set
        predpkl_test = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_Y_test_pred.pkl'
        testpkl = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_Y_test.pkl'
        prediction_test_pkl = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_prediction_test.pkl'
    else:
        # SAVE results for val set
        predpkl_test = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_Y_val_pred.pkl'
        testpkl = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_Y_val.pkl'
        prediction_test_pkl = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_prediction_val.pkl'

    (y_pred_test).to_pickle(predpkl_test)
    (y_predict_test).to_pickle(prediction_test_pkl)
    (Y_test).to_pickle(testpkl)

    return y_pred_test ,testpkl,predpkl_test, prediction_test_pkl