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


def tuning_the_model(x_val_ftp, y_val_ftp):
    """
    The function gets X set and y target
    runs gridsearch or randomsearch
    return the path str where grid_search.best_estimator_ - model with best parameters is
    """
    # Here we try to deal with imbalanced target feature in XGBoost
    # scale_pos_w =  total_negative_examples / total_positive_examples
    spw = sqrt(len(y_val_ftp[y_val_ftp == 0]) / len(y_val_ftp[y_val_ftp == 1]))
    # XGB parameters
    estimator = XGBClassifier(  # to use this function - comment/uncomment needed hyperparameters
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
    parameters = {
        'max_depth': range(2, 8, 1),
        'n_estimators': range(60, 270, 10),
        'learning_rate': np.arange(0.01, 0.3, 0.01),
        'gamma': np.arange(0.1, 9, 0.1),
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
        'reg_lambda': np.arange(0, 1., 0.2),
        'colsample_bytree': np.arange(0.5, 1., 0.1),
    }
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring='roc_auc',
        n_jobs=10,
        cv=5,
        verbose=True
    )

    grid_search.fit(x_val_ftp, y_val_ftp)
    print(grid_search.best_estimator_)

    # dump into file best model
    filename = 'Aucroc/Gap_6_' + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__best_xgb2.pkl'
    filehandler = open(filename, "wb")
    pickle.dump(grid_search.best_estimator_, filehandler)
    filehandler.close()
    return filename
