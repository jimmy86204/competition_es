"""
model trainer and optimizer
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from scipy import stats
import gc
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import pickle

def trainer(train_x, train_df, valid_x, valid_df, cat_cols, features):
    '''To build a classifer and training
    intput:
        train_x: training data with feautres
        train_df: training data with labels
        valid_x: validation data with feautres
        valid_df: validation data with labels
        cat_cols: categorical columns
        features: features name
    output: trained classifer
    '''
    lgb_params = {
        'objective' : 'binary',
        'metric' : 'average_precision',
        'learning_rate': 0.01,
        'max_depth': 4,

        'num_iterations': 5500,
        'boosting_type': 'gbdt', # dart
        'seed': 42,

        'feature_fraction': 0.50,
        'bagging_freq': 5,
        'bagging_fraction': 0.25,
        'max_bin': 100,
        'min_data_in_leaf': 100,

        'n_jobs': 10,
        'verbose': -1,
        'lambda_l2': 1,
        'lambda_l1': 1,
    }
    es = lgb.early_stopping(250, verbose=False, min_delta=1e-4)
    log = lgb.log_evaluation(period=100)
    clf = LGBMClassifier(**lgb_params)
    clf.fit(train_x[features].round(2).astype('float32'), 
            train_df['label'].reset_index(drop=True),
            categorical_feature=cat_cols,
            eval_set = [(train_x[features].round(2).astype('float32'), train_df['label'].reset_index(drop=True)),
                        (valid_x[features].round(2).astype('float32'), valid_df['label'].reset_index(drop=True))],
            callbacks=[log, es],
                )
    return clf

def optimizer_thr(preds, trues):
    '''To find the best threshold to convert probabilities into binary labels
    intput:
        preds: predication
        trues: ground truth
    output: best threshold and best f1 score
    '''
    scores = []; thresholds = []
    best_score = 0; best_threshold = 0

    for threshold in np.arange(0.0, 1.0, 0.01):
        pred = (preds>threshold).astype('int')
        m = f1_score(trues.reset_index(drop=True).values, pred)   
        scores.append(m)
        thresholds.append(threshold)
        if m>best_score:
            best_score = m
            best_threshold = threshold

    return best_threshold, best_score