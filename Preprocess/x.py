"""
data preprocess & feature engineering
"""
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy import stats
import gc
from tqdm.auto import tqdm

def merge_all_data(data):
    '''To merge all feature into a datafame
    input:
        data: a list to store all feature, which one element is a list include feature, numerical columns name and categorical columns name
    output: tuple with shape (3, )
       merged dataframe, all numerical columns name, all categorical columns name
    '''
    num_cols = [x[1] for x in data]
    cat_cols = [x[2] for x in data]
    dfs = [x[0].reset_index(drop=True) for x in data]
    return pd.concat(dfs, axis=1), [item for sublist in num_cols for item in sublist], [item for sublist in cat_cols for item in sublist]

def counting_features(history, history_all, df):
    '''To calculate using times for chid / cano level in historical data as features
    input:
        history: historical data with label = 0
        history_all : historical data with  label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    tmp_1 = history.groupby(["chid", "cano"]).size().reset_index(name="cano_counting")
    tmp_2 = history.groupby(["chid"]).size().reset_index(name="total_use_counting")
    tmp_3 = history.groupby(["chid"])["conam"].sum().reset_index(name="total_use_sum")
    tmp_4 = history_all.groupby(["chid"]).cano.nunique().reset_index(name="hold_card_number")

    merged = df[["txkey", "chid", "cano"]].merge(tmp_1, on=["chid", "cano"], how="left").fillna(0) \
                                          .merge(tmp_2, on=["chid"], how="left").fillna(0) \
                                          .merge(tmp_3, on=["chid"], how="left").fillna(0) \
                                          .merge(tmp_4, on=["chid"], how="left")
    merged = merged[["cano_counting", "total_use_counting", "total_use_sum", "hold_card_number"]]
    merged = merged.fillna(merged.mean())
    return merged[["cano_counting", "total_use_counting", "total_use_sum", "hold_card_number"]], \
                  ["cano_counting", "total_use_counting", "total_use_sum", "hold_card_number"], []

def cheat_counting_features(history, df):
    '''To calculate unauthorized use times for chid / cano level in historical data as features
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    tmp_1 = history.groupby(["chid", "cano"]).label.sum().reset_index(name="cano_cheat_counting")
    tmp_2 = history.groupby(["chid"]).label.sum().reset_index(name="cheat_counting")
    tmp_3 = history.groupby(["chid"]).size().reset_index(name="total_counting")
    merged = df[["txkey", "chid", "cano"]].merge(tmp_1, on=["chid", "cano"], how="left") \
                                          .merge(tmp_2, on=["chid"], how="left") \
                                          .merge(tmp_3, on=["chid"], how="left").fillna(0)
    merged["cano_cheat_procentage"] = merged["cano_cheat_counting"] / merged["total_counting"]
    merged["cheat_counting_procentage"] = merged["cheat_counting"] / merged["total_counting"]
    return merged[["cano_cheat_counting", 
                   "cheat_counting", 
                   "cano_cheat_procentage", 
                   "cheat_counting_procentage", 
                   "total_counting"]].fillna(0), \
           ["cano_cheat_procentage", "cheat_counting_procentage", "cano_cheat_counting", "cheat_counting", "total_counting"], []

def get_chid_num_cols_history(history, df):
    '''To calculate descriptive statistics for interested numerical columns in historical data as features
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    num_cols_history_df = history.groupby(["chid"])[num_cols].agg(["min", "mean", "max", "std", "sum"])
    num_cols_history_df.columns = [x[0] + "_" + x[1] for x in num_cols_history_df.columns]
    cols = list(num_cols_history_df.columns)
    merged = df.merge(num_cols_history_df.reset_index(), on=["chid"], how="left")
    merged[cols] = merged[cols].fillna(merged[cols].mean())
    fun_num_cols = []
    fun_cat_cols = []
    for col in num_cols:
        merged[f"{col}_mean_ratio"] = merged[col] / (merged[f"{col}_mean"]+1e-4)
        merged[f"{col}_max_large"] = (merged[col] > merged[f"{col}_max"]) * 1
        fun_num_cols.append(f"{col}_mean_ratio")
        fun_cat_cols.append(f"{col}_max_large")
    return merged[cols + fun_num_cols + fun_cat_cols], cols + fun_num_cols, fun_cat_cols

def get_chid_num_by_caon_features(history, df):
    '''To calculate the ratio of the consumer's transaction amount to the normal transaction amount based on historical data 
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    num_cols_history_df = history.groupby(["chid", "cano"])["flam1"].agg(["mean", "max"]).reset_index()
    merged = df.merge(num_cols_history_df, on=["chid", "cano"], how="left")
    cols = ["mean", "max"]
    merged[cols] = merged[cols].fillna(merged[cols].mean())
    merged["mean_ratio_cano"] = merged["flam1"] / (merged[f"mean"]+1e-4)
    merged["max_large_cano"] = (merged["flam1"] > merged[f"max"]) * 1
    return merged[["mean_ratio_cano", "max_large_cano"]], ["mean_ratio_cano", "max_large_cano"], []

def get_chid_cate_cols_history(history, df):
    '''To calculate the consumer's consumption status based on categorical data in historical records
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    cat_cols_history_df = history.groupby(["chid"])[cat_cols].agg("nunique").add_suffix("_nunique")
    use_cols = list(cat_cols_history_df.columns)
    merged = df.merge(cat_cols_history_df.reset_index(), on=["chid"], how="left").fillna(0)
    return merged[use_cols], use_cols, []
       
def last_use_features(history, df):
    '''To calculate the time difference between the authorization dates of the consumer's current transaction and their last transaction
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    df_concat = pd.concat([history, df], axis=0)
    tmp_df_list = []
    for cur_time in range(df.locdt.min(), df.locdt.max()+1):
        tmp = df_concat[(df_concat.locdt < cur_time)]
        last_day = tmp.groupby(["chid"]).locdt.last().reset_index(name="last_day")
        last_day["locdt"] = cur_time
        tmp_df_list.append(last_day)

    tmp_df = pd.concat(tmp_df_list, axis=0)
    df = df.merge(tmp_df, on=["chid", "locdt"], how="left").fillna(-10)
    df["waiting_time"] = df["locdt"] - df["last_day"]
    return df[["waiting_time"]], ["waiting_time"], []

def last_cheat_features(history, df):
    '''To calculate the time difference between the consumer's current transaction and the last occurrence of unauthorized usage
    input:
        history: historical data with label = 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    tmp = history.groupby(["chid"]).locdt.last().reset_index(name=f"last_cheat_locdt")
    df = df.merge(tmp, on=["chid"], how="left").fillna(-999)
    df[f"waiting_cheat_time"] = df["locdt"] - df[f"last_cheat_locdt"]
    return df[["waiting_cheat_time"]], ["waiting_cheat_time"], []

def use_frequency_features(history, df):
    '''To calculate the consumer's spending frequency
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    x = history.drop_duplicates(["chid", "locdt"])[["chid", "locdt"]]
    x["locdt_1"] = x.groupby(["chid"]).locdt.shift(1)
    x["diff"] = x["locdt"] - x["locdt_1"]
    p = x.groupby(["chid"])["diff"].mean().reset_index(name="use_frequency")
    df = df[["chid"]].merge(p, on=["chid"], how="left").fillna(60)
    return df[["use_frequency"]], ["use_frequency"], []
    
def even_use_features(history, df):
    '''To calculate spending habits under specific conditions
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_cols = []
    for col in hobby_features:
        tmp = history.groupby(["chid"])[col].value_counts(normalize=True).reset_index(name=f"{col}_proportion")
        df = df.merge(tmp, on=["chid", col], how="left").fillna(0)
        use_cols.append(f"{col}_proportion")
    return df[use_cols], use_cols, []

def important_features(history, df):
    '''To calculate the probability of unauthorized usage under specific conditions and provide descriptive statistics
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_col = []
    for col in important_cols:
        tmp = history.groupby([col]).label.mean().reset_index(name=f"{col}_important_hobby_mean")
        tmp2 = history.groupby([col]).label.std().reset_index(name=f"{col}_important_hobby_std")
        tmp3 = history.groupby([col]).label.sum().reset_index(name=f"{col}_important_hobby_sum")
        df = df.merge(tmp, on=[col], how="left").fillna(0) \
               .merge(tmp2, on=[col], how="left").fillna(0) \
               .merge(tmp3, on=[col], how="left").fillna(0)
        use_col.append(f"{col}_important_hobby_mean")
        use_col.append(f"{col}_important_hobby_std")
        use_col.append(f"{col}_important_hobby_sum")
    return df[use_col], use_col, []

def important_by_weekday_features(history, df):
    '''To calculate the probability of unauthorized usage under specific conditions with weekday and provide descriptive statistics
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_col = []
    for col in important_cols:
        tmp = history.groupby([col, "weekday"]).label.mean().reset_index(name=f"{col}_weekday_important_hobby_mean")
        df = df.merge(tmp, on=[col, "weekday"], how="left")
        df[f"{col}_weekday_important_hobby_mean"] = df[f"{col}_weekday_important_hobby_mean"].fillna(df[f"{col}_weekday_important_hobby_mean"].mean())
        use_col.append(f"{col}_weekday_important_hobby_mean")

    return df[use_col], use_col, []

# def hour_hobby_features(history, df):
#     '''To calculate the user's spending habits per hour
#     input:
#         history: historical data with label = 0
#         df: training data
#     output: tuple with shape (3, )
#        processed dataframe, numerical columns name: list, categorical columns name: list
#     '''
#     use_col = []
#     for col in important_cols:
#         tmp = history.groupby([col])["hour"].value_counts(normalize=True).reset_index(name=f"{col}_hour_hobby")
#         df = df.merge(tmp, on=[col, "hour"], how="left").fillna(0)
#         use_col.append(f"{col}_hour_hobby")
#     return df[use_col], use_col, []

def hour_hobby_features(history, df):
    '''To calculate the user's spending habits per hour
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_col = []
    for col in important_cols:
        for time_cols in ["hour", "weekday"]:
            tmp = history.groupby([col])[time_cols].value_counts(normalize=True).reset_index(name=f"{col}_{time_cols}_hobby")
            df = df.merge(tmp, on=[col, time_cols], how="left").fillna(0)
            use_col.append(f"{col}_{time_cols}_hobby")
    return df[use_col], use_col, []

def last_cheat_cano_date(history, df):
    '''To calculate the characteristics of credit card fraud
    input:
        history: historical data with label = 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    tmp  = history.groupby(["cano"]).locdt.last().reset_index(name="last_cheat_day")
    tmp2 = history.groupby(["cano"]).locdt.nunique().reset_index(name="nunique_cheat_day")
    tmp3 = history.groupby(["cano"]).locdt.value_counts().reset_index().groupby(["cano"])["count"].sum().reset_index(name="sum_cheat")
    tmp4 = history.groupby(["cano"]).locdt.value_counts().reset_index().groupby(["cano"])["count"].max().reset_index(name="max_cheat")
    df = df.merge(tmp, on=["cano"], how="left").fillna(-999) \
           .merge(tmp2, on=["cano"], how="left").fillna(0) \
           .merge(tmp3, on=["cano"], how="left").fillna(0) \
           .merge(tmp4, on=["cano"], how="left").fillna(0)
    return df[["last_cheat_day", "nunique_cheat_day", "sum_cheat", "max_cheat"]], ["last_cheat_day", "nunique_cheat_day", "sum_cheat", "max_cheat"], []
    
def money_diff_with_history(history, df):
    '''To calculate statistical measures and proportions of transaction amounts under specific conditions
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_cols = []
    for col in money_history_cols:
        tmp = history.groupby(col)["conam"].agg(["mean", "max"])
        tmp.columns = [f"{col}_{x}_money_history" for x in tmp.columns]
        cols = list(tmp.columns)
        use_cols += cols
        df = df.merge(tmp, on=[col], how="left")
        for c in cols:
            if c in ["stocn_sum_money_history", "mcc_sum_money_history"]:
                continue
            if "std" in c:
                continue
            df[c] = df[c].fillna(df[c].mean())
            df[f"{c}_ratio"] = df["conam"] / (df[c]+1e-4)
            use_cols.append(f"{c}_ratio")
    return df[use_cols], use_cols, []

def money_diff_with_cheat_history(history, df):
    '''To calculate the similarity between transaction amounts under specific conditions and fraudulent transaction amounts
    input:
        history: historical data with label = 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_cols = []
    for col in money_history_cols:
        tmp = history.groupby(col)["conam"].agg(["mean", "max"])
        tmp.columns = [f"{col}_{x}_money_history_cheat" for x in tmp.columns]
        cols = list(tmp.columns)
        df = df.merge(tmp, on=[col], how="left")
        for c in cols:
            df[c] = df[c].fillna(df[c].mean())
            df[f"{c}_ratio"] = df["conam"] / (df[c]+1e-4)
            use_cols.append(f"{c}_ratio")
    return df[use_cols], use_cols, []

def train_imformation(history, df):
    '''To calculate the similarity between occurrences and frequency of occurrences in a specific field for a time during the training data cycle and those in past normal transactions
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_cols = ["one_day_counts_hobby_diff", "one_day_chid_counts",
                "one_day_mcc_counts", "one_day_mchno_counts",
                "one_day_scity_counts", "one_day_stocn_counts",
                "one_day_csmam_counts"]

    ref = history.groupby(["chid", "locdt"]).size().reset_index(name="one_day_counts").groupby(["chid"]).one_day_counts.mean().reset_index()    
    tmp1 = df.groupby(["chid", "locdt"]).size().reset_index(name=f"one_day_chid_counts")
    tmp2 = df.groupby(["chid", "locdt"]).mcc.nunique().reset_index(name=f"one_day_mcc_counts")
    tmp3 = df.groupby(["chid", "locdt"]).mchno.nunique().reset_index(name=f"one_day_mchno_counts")
    tmp4 = df.groupby(["chid", "locdt"]).scity.nunique().reset_index(name=f"one_day_scity_counts")
    tmp5 = df.groupby(["chid", "locdt"]).stocn.nunique().reset_index(name=f"one_day_stocn_counts")
    tmp6 = df.groupby(["chid", "locdt"]).csmam.nunique().reset_index(name=f"one_day_csmam_counts")
    
    df = df.merge(tmp1, on=["chid", "locdt"], how="left") \
           .merge(tmp2, on=["chid", "locdt"], how="left") \
           .merge(tmp3, on=["chid", "locdt"], how="left") \
           .merge(tmp4, on=["chid", "locdt"], how="left") \
           .merge(tmp5, on=["chid", "locdt"], how="left") \
           .merge(tmp6, on=["chid", "locdt"], how="left")
    
    df = df.merge(ref, on=["chid"], how="left")
    
    df["one_day_counts"] = df["one_day_counts"].fillna(df["one_day_counts"].mean())
    df["one_day_counts_hobby_diff"] = df["one_day_chid_counts"] / (df["one_day_counts"])
    return df[use_cols], use_cols, []

def period_information(df, day_len):
    '''To calculate the frequency of occurrences in a specific field during the training data cycle
    input:
        df: training data
        day_len: training data time period length
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    tmp = df.groupby(["chid"]).conam.std().reset_index(name="period_conam_std")
    tmp["period_conam_std"] = tmp["period_conam_std"].fillna(tmp["period_conam_std"].mean())
    tmp2 = (df.groupby(["chid"]).mcc.nunique() / day_len).reset_index(name="period_mcc_nunique")
    tmp3 = (df.groupby(["chid"]).mchno.nunique() / day_len).reset_index(name="period_mchno_nunique")
    tmp4 = (df.groupby(["chid"]).scity.nunique() / day_len).reset_index(name="period_scity_nunique")
    tmp5 = (df.groupby(["chid"]).stocn.nunique() / day_len).reset_index(name="period_stocn_nunique")
    tmp6 = (df.groupby(["chid"]).size() / day_len).reset_index(name="period_size")
    
    df = df.merge(tmp, on=["chid"], how='left') \
           .merge(tmp2, on=["chid"], how='left') \
           .merge(tmp3, on=["chid"], how='left') \
           .merge(tmp4, on=["chid"], how='left') \
           .merge(tmp5, on=["chid"], how='left') \
           .merge(tmp6, on=["chid"], how='left')
    
    df["period_mcc_nunique"] = df["period_mcc_nunique"] / df["period_size"]
    df["period_mchno_nunique"] = df["period_mchno_nunique"] / df["period_size"]
    df["period_scity_nunique"] = df["period_scity_nunique"] / df["period_size"]
    df["period_stocn_nunique"] = df["period_stocn_nunique"] / df["period_size"]
    return df[["period_conam_std", "period_mcc_nunique", "period_mchno_nunique", "period_scity_nunique", "period_stocn_nunique", "period_size"]], ["period_conam_std", "period_mcc_nunique", "period_mchno_nunique", "period_scity_nunique", "period_stocn_nunique", "period_size"], []
   
def important_prev_period_features(history, df):
    '''To calculate the characteristics of the most recent fraudulent occurrence in a specific field
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_col = []
    period = 5 # 4 or 5
    curr_period = df.locdt.min()
    prev_period_df = history[history.locdt >= curr_period - period]
    for col in important_cols:
        tmp = prev_period_df.groupby([col]).label.mean().reset_index(name=f"{col}_prev_period_important_hobby_mean")
        tmp2 = prev_period_df.groupby([col]).label.std().reset_index(name=f"{col}_prev_period_important_hobby_std")
        tmp3 = prev_period_df.groupby([col]).label.sum().reset_index(name=f"{col}_prev_period_important_hobby_sum")
        df = df.merge(tmp, on=[col], how="left").fillna(0) \
               .merge(tmp2, on=[col], how="left").fillna(0) \
               .merge(tmp3, on=[col], how="left").fillna(0)
        use_col.append(f"{col}_prev_period_important_hobby_mean")
        use_col.append(f"{col}_prev_period_important_hobby_std")
        use_col.append(f"{col}_prev_period_important_hobby_sum")
    return df[use_col], use_col, []

def mchno_and_mcc_features(history, df):
    '''To calculate the descriptive statistics of transaction amounts for a specific field on a given day
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    tmp1 = df.groupby(["locdt", "mchno"]).csmam.skew().reset_index(name="macho_skew")
    tmp2 = df.groupby(["locdt", "mchno"]).csmam.mean().reset_index(name="macho_mean")
    tmp3 = df.groupby(["locdt", "mchno"]).size().reset_index(name="one_day_macho_size")
    tmp4 = df.groupby(["locdt", "mcc"]).size().reset_index(name="one_day_mcc_size")
    tmp5 = history.groupby(["mchno"]).size().reset_index(name="mchno_history_mean_size")
    tmp6 = history.groupby(["mchno"]).csmam.mean().reset_index(name="mchno_history_mean_csmam")
    
    tmp1.macho_skew = tmp1.macho_skew.fillna(tmp1.macho_skew.mean())

    df = df.merge(tmp3, on=["locdt", "mchno"], how="left") \
           .merge(tmp4, on=["locdt", "mcc"], how="left") \
           .merge(tmp5, on=["mchno"], how="left") \
           .merge(tmp1, on=["locdt", "mchno"], how="left") \
           .merge(tmp2, on=["locdt", "mchno"], how="left") \
           .merge(tmp6, on=["mchno"], how="left") \
    
    df.mchno_history_mean_size = df.mchno_history_mean_size.fillna(df.mchno_history_mean_size.mean())
    df.mchno_history_mean_csmam = df.mchno_history_mean_csmam.fillna(df.mchno_history_mean_csmam.mean())
    df["mchno_size_ratio"] = df["one_day_macho_size"] / df["mchno_history_mean_size"]
    df["mchno_csmam_ratio"] = df["macho_mean"] / df["mchno_history_mean_csmam"]
    df["mchno_csmam_ratio"] = df["mchno_csmam_ratio"].fillna(df["mchno_csmam_ratio"].mean())
    return df[["one_day_macho_size", "one_day_mcc_size", "mchno_size_ratio", "macho_skew", "mchno_csmam_ratio"]], ["one_day_macho_size", "one_day_mcc_size", "mchno_size_ratio", "macho_skew", "mchno_csmam_ratio"], []

def csmam_rank_pct(df_):
    '''To calculate the percentile of the transaction amount among all transactions today
    input:
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    df = df_.copy()
    df["csmam_rank_pct"] = df.groupby(["locdt", "mchno"]).csmam.rank(method="min", pct=True)
    return df[["csmam_rank_pct"]], ["csmam_rank_pct"], []
    
def prev_time_information(history, df):
    '''To calculate the characteristics of the most last fraudulent occurrence in a specific field
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_cols = []
    min_locdt = df.locdt.min()
    history = history[history.locdt == min_locdt - 1]
    for col in important_cols:
        tmp1 = history.groupby([col]).label.mean().reset_index(name=f"prev_{col}_cheat_mean")
        tmp2 = history.groupby([col]).label.sum().reset_index(name=f"prev_{col}_cheat_sum")
        tmp3 = history.groupby([col]).label.std().reset_index(name=f"prev_{col}_cheat_std")
        df = df.merge(tmp1, on=[col], how="left") \
               .merge(tmp2, on=[col], how="left") \
               .merge(tmp3, on=[col], how="left")
        df[f"prev_{col}_cheat_mean"] = df[f"prev_{col}_cheat_mean"].fillna(df[f"prev_{col}_cheat_mean"].mean())
        df[f"prev_{col}_cheat_sum"] = df[f"prev_{col}_cheat_sum"].fillna(df[f"prev_{col}_cheat_sum"].mean())
        df[f"prev_{col}_cheat_std"] = df[f"prev_{col}_cheat_std"].fillna(df[f"prev_{col}_cheat_std"].mean())
        use_cols.append(f"prev_{col}_cheat_mean")
        use_cols.append(f"prev_{col}_cheat_sum")
        use_cols.append(f"prev_{col}_cheat_std")
    return df[use_cols], use_cols, []

def curr_vs_normal(history, df):
    '''To calculate the ratio of the number of transactions in a specific field to the past data
    input:
        history: historical data with label = 0
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    use_cols = []
    for col in important_cols:
        tmp1 = df.groupby([col, "locdt"]).size().reset_index(name=f"{col}_one_day_size")
        tmp2 = (history.groupby([col]).size() / history.locdt.max()) .reset_index(name=f"{col}_history_day_size")
        df = df.merge(tmp1, on=["locdt", col], how="left") \
               .merge(tmp2, on=[col], how="left")
        df[f"{col}_history_day_size"] = df[f"{col}_history_day_size"].fillna(df[f"{col}_history_day_size"].mean())
        df[f"{col}_day_size_gap"] = df[f"{col}_one_day_size"] / df[f"{col}_history_day_size"]
        use_cols.append(f"{col}_day_size_gap")
    return df[use_cols], use_cols, []

def is_last_use(history, df):
    '''To calculate whether the credit card is the last transaction within a time cycle
    input:
        history: historical data with label = 0, 1
        df: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    df_concat = pd.concat([history, df], axis=0)
    tmp = df_concat.groupby(["cano"]).locdt.last().reset_index(name="last_use_day")
    df = df.merge(tmp, on=["cano"], how="left")
    df["is_last_use_day"] = (df["last_use_day"] == df["locdt"]) * 1
    df["gap_to_last"] = df.locdt.max() - df["locdt"]
    return df[["is_last_use_day", "gap_to_last"]], ["gap_to_last"], ["is_last_use_day"]

def oversea(history_, df_):
    '''To calculate statistical measures of oversea transactions
    input:
        history_: historical data with label = 0, 1
        df_: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    history = history_.copy()
    df = df_.copy()
    history_oversea = history.loc[history['is_oversea']==1]

    tmp1 = history.groupby(["chid"]).is_oversea.sum().reset_index(name="oversea_sum")
    tmp2 = history.groupby(["chid"]).is_oversea.mean().reset_index(name="oversea_mean")
    tmp3 = history_oversea.groupby(["chid"]).conam.mean().reset_index(name="oversea_money_mean")
    tmp4 = history_oversea.groupby(["chid"]).conam.max().reset_index(name="oversea_money_max")

    df = df.merge(tmp1, on=["chid"], how="left").fillna(0) \
           .merge(tmp2, on=["chid"], how="left").fillna(0) \
           .merge(tmp3, on=["chid"], how="left").fillna(0) \
           .merge(tmp4, on=["chid"], how="left").fillna(0)

    df["diff_oversea_money_mean"]  = df["is_oversea"] * (df["oversea_money_mean"] - df["conam"])
    df["oversea_money_mean_large"] = df["is_oversea"] * (df["conam"] > df["oversea_money_max"])*1

    return df[["oversea_sum", "oversea_mean", "diff_oversea_money_mean", "oversea_money_mean_large"]], ["oversea_sum", "oversea_mean", "diff_oversea_money_mean", "oversea_money_mean_large"], []

def tag_proportion_features(history_, df_):
    '''To calculate proportions of specific conditions in normal transactions
    input:
        history_: historical data with label = 0
        df_: training data
    output: tuple with shape (3, )
       processed dataframe, numerical columns name: list, categorical columns name: list
    '''
    history = history_.copy()
    df = df_.copy()

    use_cols = []
    for col in ['ecfg', 'flg_3dsmk', 'etymd_4', 'stocn_twn']:
        tmp = history.groupby(["mchno"])[col].value_counts(normalize=True).reset_index(name=f"mchno_{col}_proportion")
        df = df.merge(tmp, on=["mchno", col], how="left").fillna(0)
        use_cols.append(f"mchno_{col}_proportion")
    return df[use_cols], use_cols, []


def feature_engineering(df, history_all, history_0, history_1, train_len):
    '''To do all feature engineering
    input:
        df: data
        history_all: historical data with label = 0, 1
        history_0: historical data with label = 0
        history_1: historical data with label = 1
        train_len: data time period length
    output: tuple with shape (3, )
       processed data with all features, numerical columns, categorical columns
    '''
    train_df = df.copy()
    history_train = history_all.copy()
    history_train_label_0 = history_0.copy()
    history_train_label_1 = history_1.copy()

    print("is_last_use")
    train_is_last_use = is_last_use(history_train, train_df)

    print("curr_vs_normal")
    train_curr_vs_normal = curr_vs_normal(history_train_label_0, train_df)

    print("prev features")
    train_prev_time_information = prev_time_information(history_train, train_df)

    print("skew_and_kurtosis")
    train_skew_and_kurtosis_features_df = mchno_and_mcc_features(history_train_label_0, train_df)

    print("csmam_rank_pct")
    train_csmam_rank_pct_features_df = csmam_rank_pct(train_df)

    print("crazy features")
    train_crazy_features_df = period_information(train_df, train_len)

    print("train_imformation")
    train_features_df = train_imformation(history_train_label_0, train_df)

    print("cheat with card")
    train_cheat_with_card_features_df = last_cheat_cano_date(history_train_label_1, train_df)

    print("cheat money")
    train_features_history_money_cheat = money_diff_with_cheat_history(history_train_label_1, train_df)

    print("history money")
    train_features_history_money = money_diff_with_history(history_train_label_0, train_df)

    print("cat")
    train_features_cate = train_df[use_cat_cols_list], [], use_cat_cols_list

    print("num")
    train_features_num = train_df[num_cols], num_cols, []

    print("no_process_num")
    train_features_no_process_num = train_df[no_process_num], no_process_num, []

    print("cheat counting")
    train_features_counting_cheat = cheat_counting_features(history_train, train_df)

    print("counting")
    train_features_counting = counting_features(history_train_label_0, history_train, train_df)

    print("user num")
    train_features_user_num = get_chid_num_cols_history(history_train_label_0, train_df)

    print("user num cano")
    train_features_user_num_cano = get_chid_num_by_caon_features(history_train_label_0, train_df)

    print("user cat")
    train_features_user_cate = get_chid_cate_cols_history(history_train_label_0, train_df)

    print("last feature")
    train_features_user_last = last_use_features(history_train, train_df)

    print("use frequency feature")
    train_features_use_frequency = use_frequency_features(history_train_label_0, train_df)

    print("hobby feature")
    train_features_user_hobby = even_use_features(history_train_label_0, train_df)

    print("hobby important hobby")
    train_important_hobby = important_features(history_train, train_df)

    print("hobby prev period important hobby")
    train_prev_period_important_hobby = important_prev_period_features(history_train, train_df)
    
    print("is end with 99")
    train_features_is_99 = train_df.conam.apply(lambda x: (x % 100 == 99) * 1).reset_index(name="is_99")[["is_99"]], [], ["is_99"]

    print("is % 100")
    train_features_is_100 = train_df.conam.apply(lambda x: (x % 100 == 0) * 1).reset_index(name="is_100")[["is_100"]], [], ["is_100"]

    print("is % 10")
    train_features_is_10 = train_df.conam.apply(lambda x: (x % 10 == 0) * 1).reset_index(name="is_10")[["is_10"]], [], ["is_10"]

    print("is 0")
    train_features_is_0 = train_df.conam.apply(lambda x: (x == 0) * 1).reset_index(name="is_0")[["is_0"]], [], ["is_0"]

    print("last cheat feature")
    train_last_cheat_feature_df = last_cheat_features(history_train_label_1, train_df)

    print("hour hobby")
    train_hour_hobby_features_df = hour_hobby_features(history_train_label_0, train_df)

    print("oversea")
    train_oversea = oversea(history_train, train_df)

    print("tag proportion")
    train_tag_proportion_features = tag_proportion_features(history_train_label_0, train_df)

    print("merge all data")
    all_train_features = [train_features_cate, train_features_num, train_features_no_process_num, train_prev_time_information, train_curr_vs_normal,
                        train_features_counting, train_features_user_num, train_features_user_cate, train_features_is_100, train_features_is_99, train_features_is_10, train_features_is_0,
                        train_features_user_last, train_features_user_hobby, train_last_cheat_feature_df, train_features_counting_cheat,
                        train_hour_hobby_features_df, train_features_use_frequency, train_features_user_num_cano, train_prev_period_important_hobby,
                        train_features_history_money, train_features_history_money_cheat, train_crazy_features_df, train_csmam_rank_pct_features_df,
                        train_cheat_with_card_features_df, train_features_df, train_important_hobby, train_skew_and_kurtosis_features_df, train_is_last_use, train_oversea, train_tag_proportion_features]

    
    train_x, use_num_cols, use_cat_cols = merge_all_data(all_train_features)
    return train_x, use_num_cols, use_cat_cols

if __name__ == "__main__":
    train1 = pd.read_csv('./Data/training_data/training.csv') # load training data
    train2 = pd.read_csv('./Data/training_data/public.csv') # load training data
    train = pd.concat([train1, train2], axis=0)
    test = pd.read_csv('./Data/training_data/private_1_processed.csv') # load testing data
    ori_test = test.copy()
    del train1, train2
    gc.collect()
    train["weekday"] = train.locdt % 7
    train["hour"] = train.loctm.apply(lambda x: int("{:06d}".format(x)[:2]))
    train["hour_6"] = train["hour"] // 6
    train['is_oversea'] = (((train['stocn']==0) & (train['csmcu']!=70)) | ((train['stocn']!=0) & (train['csmcu']==70))).astype(int)
    train['etymd_4'] = 0
    train.loc[train['etymd'] == 4, 'etymd_4'] = 1
    train['stocn_twn'] = 0
    train.loc[train['stocn'] == 0, 'stocn_twn'] = 1
    train = train.sort_values(by=["locdt"]).reset_index(drop=True)

    test["weekday"] = test.locdt % 7
    test["hour"] = test.loctm.apply(lambda x: int("{:06d}".format(x)[:2]))
    test["hour_6"] = test["hour"] // 6
    test['is_oversea'] = (((test['stocn']==0) & (test['csmcu']!=70)) | ((test['stocn']!=0) & (test['csmcu']==70))).astype(int)
    test['etymd_4'] = 0
    test.loc[test['etymd'] == 4, 'etymd_4'] = 1
    test['stocn_twn'] = 0
    test.loc[test['stocn'] == 0, 'stocn_twn'] = 1
    test = test.sort_values(by=["locdt"]).reset_index(drop=True)

    important_cols = ["mcc", "mchno", "chid", "stocn", "scity"]
    use_cat_cols_list = ["etymd", "mcc", "ecfg", "stocn", "stscd", "hcefg", "flg_3dsmk"]
    cat_cols = ["contp", "etymd", "mcc", "ecfg", "insfg", "bnsfg", "stocn", "stscd", "ovrlt", "flbmk", "hcefg", "csmcu", "flg_3dsmk", "weekday"]
    num_cols = ["flam1"]
    precoess_cols = ["mchno", "stocn", "scity", "mcc"]
    no_process_num = ["hour", "hour_6"]
    hobby_features = ["mchno", "mcc", "scity", "ecfg", "stocn"] + ["weekday"]
    even_buy_cols = ["mcc", "stocn", "scity", "mchno"]
    money_history_cols = ["mcc", "mchno", "stocn", "scity"]

    train.loc[:, cat_cols] = train[cat_cols].fillna(-999)
    train.loc[:, precoess_cols] = train[precoess_cols].fillna(-999)
    le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train.loc[:, cat_cols] = le.fit_transform(train[cat_cols])

    test.loc[:, cat_cols] = test[cat_cols].fillna(-999)
    test.loc[:, precoess_cols] = test[precoess_cols].fillna(-999)
    test.loc[:, cat_cols] = le.transform(test[cat_cols])

    output = open(f'./Data/le.pkl', 'wb')
    pickle.dump(le, output)
    output.close()

    K = 5
    train_len = 12
    valid_len = 5
    start_date = train.locdt.max() - (valid_len * (K-0)) - train_len
    oof_predictions = pd.DataFrame()
    for k in tqdm(range(K)):
        start_date = train.locdt.max() - (valid_len * (K-k)) - train_len
        
        history_train = train[(train.locdt <= start_date)]
        history_valid = train[(train.locdt <= start_date + train_len)]
        history_train = history_train.groupby('chid').tail(20)
        history_valid = history_valid.groupby('chid').tail(20)
        history_train_label_0 = history_train[history_train.label == 0].reset_index(drop=True)
        history_valid_label_0 = history_valid[history_valid.label == 0].reset_index(drop=True)
        history_train_label_1 = history_train[history_train.label == 1].reset_index(drop=True)
        history_valid_label_1 = history_valid[history_valid.label == 1].reset_index(drop=True)
        train_df = train[(train.locdt <= start_date + train_len) & (train.locdt >= start_date + 1)]
        valid_df = train[(train.locdt <= start_date + train_len + valid_len) & (train.locdt >= start_date + train_len + 1)]
        
        train_x, use_num_cols, use_cat_cols = feature_engineering(train_df, history_train, history_train_label_0, history_train_label_1, train_len)
        valid_x, _, __ = feature_engineering(valid_df, history_valid, history_valid_label_0, history_valid_label_1, valid_len)

        output = open(f'./Data/train_{k}.pkl', 'wb')
        pickle.dump({
            "x": train_x,
            "df": train_df,
            "num_cols": use_num_cols,
            "cat_cols": use_cat_cols,
        }, output)
        output.close()

        output = open(f'./Data/valid_{k}.pkl', 'wb')
        pickle.dump({
            "x": valid_x,
            "df": valid_df,
        }, output)
        output.close()
    
    history_test = train.copy()
    history_test = history_test.groupby('chid').tail(20)
    history_test_label_0 = history_test[history_test.label == 0].reset_index(drop=True)
    history_test_label_1 = history_test[history_test.label == 1].reset_index(drop=True)
    test_df = test.copy()
    test_x, use_num_cols, use_cat_cols = feature_engineering(test_df, history_test, history_test_label_0, history_test_label_1, valid_len)
    output = open(f'./Data/test.pkl', 'wb')
    pickle.dump({
        "x": test_x,
        "df": test_df,
        "ori": ori_test,
        "num_cols": use_num_cols,
        "cat_cols": use_cat_cols,
    }, output)
    output.close()
