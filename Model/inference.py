"""
inference function
"""
def inference(df, features, clf):
    '''To execute inference
    intput:
        df: data to infer
        features: use columns
        clf: trained classifer
    output: prediction probability
    '''
    preds = clf.predict_proba(df[features].round(2).astype('float32'))
    preds = [x[1] for x in preds]
    return preds