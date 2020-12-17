# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from datetime import datetime as dt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
# from sklearn.model_selection import cross_val_score
from Vent._OnSet_Sheba_FalseP_detection._Baselines_Intervention_Prediction_Mechanical_Ventilation import *
from math import sqrt
import pickle

# from utils._FPdetection_AUCROC import Gntbok1

# import packages for hyperparameters tuning
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from utils._FPdetection_AUCROC import Gntbok1

now = dt.now()
runtime = now.strftime('%d-%m-%Y_%H%M')
RANDOM = 34000







def second_model_run(testpkl, x_test_new, x_val_ftp, y_val_ftp):
    """

    """

    y_test = pd.read_pickle(testpkl).to_numpy()
    # print("GridSearch is started")
    # best_model_path = tuning_the_model(x_val_ftp, y_val_ftp)
    # best_model_path =
    # file = open(best_model_path, 'rb')
    # object_file = pickle.load(file)
    # file.close()
    # # print("GridSearch is finished")

    # param_comb = 5
    # defining a new xgboost model
    spw = sqrt(len(y_val_ftp[y_val_ftp == 0]) / len(y_val_ftp[y_val_ftp == 1]))

    xgb2 = XGBClassifier(  # to use this function - comment/uncomment needed hyperparameters
        objective='binary:logistic',
        scale_pos_weight=spw,
        seed=RANDOM,
        gamma=0.4,
        learning_rate=0.1,
        n_estimators=120,
        max_depth=2,
        reg_lambda=0.8,
        reg_alpha=0.05,
        colsample_bytree=0.7
    )
    print(f"best model is with params\n{xgb2}")

    print('Fit on the trainingsdata')
    xgb2.fit(x_val_ftp, y_val_ftp, eval_metric='auc')
    print('Overall AUC:', roc_auc_score(y_val_ftp, xgb2.predict_proba(x_val_ftp)[:, 1]))

    # making prediction on test set with 2nd xgb model
    test_prd_xgb2 = pd.DataFrame(xgb2.predict_proba(np.nan_to_num(x_test_new)))
    test_predict_xgb2 = pd.DataFrame(xgb2.predict(np.nan_to_num(x_test_new)))

    # saving results into pkl files
    test_prd_xgb2_pkl = 'Aucroc/Gap 6 ' + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__test_pred_xgb2.pkl'
    test_prd_xgb2.to_pickle(test_prd_xgb2_pkl)
    predict_xgb2 = 'Aucroc/Gap 6 ' + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__prediction_xgb2.pkl'
    test_predict_xgb2.to_pickle(predict_xgb2)

    # evaluating 2nd xgboost model
    x_val_xgb2, y_val_xgb2, _ = Gntbok1(x_test_new, test_prd_xgb2_pkl, testpkl, predict_xgb2, 6)

    print('Evaluating 2nd xgboost model is finished')

    # check the results with StratifiedKFold
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM)
    print('Start cross validation')
    results = cross_val_score(xgb2, x_val_ftp, y_val_ftp, cv=skf)
    print("Mean Kfold Accuracy for Validation set: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
