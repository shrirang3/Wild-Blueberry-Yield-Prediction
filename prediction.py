import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression


model=joblib.load('./Notebook/blueberry_yield_model.pkl')

def select_features_infogain_based(X_train, y_train, X_test, x='all'):
    if type(x)==str:
        fs_info=SelectKBest(score_func=mutual_info_regression, k='all')
    else:
        fs_info=SelectKBest(score_func=mutual_info_regression, k=x)

    fs_info.fit(X_train, y_train)
    X_train_fs=fs_info.transform(X_train)
    X_test_fs=fs_info.transform(X_test)

    return X_train_fs, X_test_fs, fs_info

def get_prediction(model, df):
    prediction=model.predict(df)
    return prediction
    