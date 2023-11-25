'''
main process
'''
from Model.train import trainer, optimizer_thr
from Model.inference import inference
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy import stats
import gc
from tqdm.auto import tqdm
import joblib

def read_train_and_valid(k):
    '''To read train and validation data
    input:
        k: fold
    output: train and validation data
    '''
    file = open(f'./Data/train_{k}.pkl', "rb")
    train = pickle.load(file)
    file.close()

    file = open(f'./Data/valid_{k}.pkl', "rb")
    valid = pickle.load(file)
    file.close()
    return train, valid

def read_test():
    '''To read testing data
    input: nothing
    output: testing data
    '''
    file = open(f'./Data/test.pkl', "rb")
    test = pickle.load(file)
    file.close()
    return test

if __name__ == "__main__":
    K = 5
    oof_predictions = pd.DataFrame()
    for k in range(K):
        train, valid = read_train_and_valid(k)
        train_x, train_df, valid_x, valid_df, use_cat_cols = train["x"], train["df"], valid["x"], valid["df"], train["cat_cols"]
        features = train["num_cols"] + train["cat_cols"]
        del train, valid
        gc.collect()

        print("staring training")
        clf = trainer(train_x, train_df, valid_x, valid_df, use_cat_cols, features)
        joblib.dump(clf, f'./Data/lgb_{k}.pkl')

        print("staring inference")
        preds = inference(valid_x, features, clf)
        valid_df["preds"] = preds
        valid_df["fold"] = k
        oof_predictions = pd.concat([oof_predictions, valid_df[["preds", "txkey", "label", "fold"]]], axis=0)

    oof_predictions.to_csv(f'./Data/oof.csv', index=False)
    preds = oof_predictions.preds
    label = oof_predictions.label
    best_threshold, best_score = optimizer_thr(preds, label)
    print(best_threshold, best_score)

    output = open(f'./Data/thr.pkl', 'wb')
    pickle.dump({
        "best_threshold": best_threshold,
        "best_score": best_score
    }, output)
    output.close()

    del train_x, train_df, valid_x, valid_df, use_cat_cols
    gc.collect()

    test = read_test()
    test_x = test["x"]
    test_df = test["df"]
    features = test["num_cols"] + test["cat_cols"]
    for k in tqdm(range(0, K)):
                
        clf = joblib.load(f'./Data/lgb_{k}.pkl')
        print("staring pred")
        preds = clf.predict_proba(test_x[features].round(2).astype('float32'))
        preds = [x[1] for x in preds]
        test_df[f"pred_{k}"] = preds
    
    file = open(f'./Data/thr.pkl', "rb")
    thr = pickle.load(file)
    file.close()
    best_threshold = thr["best_threshold"]
    test_df["pred_mean"] = test_df[["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]].mean(axis=1)
    test_df["final"] = (test_df["pred_mean"].fillna(0) > best_threshold).astype(int)
    sub = pd.read_csv('./Data/training_data/31_範例繳交檔案.csv')
    sub.loc[:, "pred"] = 0
    merged = sub.merge(test_df[["txkey", "final"]], on=["txkey"], how="left")[["txkey", "final"]].rename(columns={"final": "pred"}).to_csv('./Output/sub.csv', index=False)