import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def scale_train_test_split(X,y, set_seed=101):
    """
        Get predictors and target and return training and test splits
    """
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    target = y # Need to modify depending on target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = set_seed)
    return X_train, X_test, y_train, y_test

def train_model(X_train,y_train):
    """ 
        Fit several models
    """
    model_names = []
    models = []
    # Adaboost
    model_names.append('AdaBoost')
    adaboost_clf = AdaBoostClassifier()
    adaboost_clf.fit(X_train, y_train)
    models.append(adaboost_clf)
    
    # GradientBoost
    model_names.append('GradientBoost')
    gbt_clf = GradientBoostingClassifier()
    gbt_clf.fit(X_train, y_train)
    models.append(gbt_clf)
    
    # # XGBoost
    # model_names.append('XGBoost')
    # xgb_clf = xgb.XGBClassifier()
    # xgb_clf.fit(X_train, y_train)
    # models.append(xgb_clf)
    
    # Logistic Regression
    model_names.append('Logistic Regresion')
    lr_clf = LogisticRegression(fit_intercept = False, C = 1e12)
    lr_clf.fit(X_train, y_train)
    models.append(lr_clf)
    
    # Decision Tree
    # model_names.append('DecisionTree_gini')
    # dtree_clf = DecisionTreeClassifier(criterion='gini')
    # dtree_clf.fit(X_train, y_train)
    # models.append(dtree_clf)

    # model_names.append('DecisionTree_entropy')
    # dtree_clf2 = DecisionTreeClassifier(criterion='entropy')
    # dtree_clf2.fit(X_train, y_train)
    # models.append(dtree_clf2)


    return models, model_names

def predict_all(x1, y1, models):
    predictions = []
    for model in models:  
        predictions.append(model.predict(x1))
    
    return predictions
    
def display_acc_and_f1_score(true, preds, model_name):
    acc = accuracy_score(true, preds)
    f1 = f1_score(true,preds, average='micro')
    print("Model: {}".format(model_name))
    print("Accuracy: {}".format(acc))
    print("F1-Score: {}".format(f1))
    return None

def others(y_test, model, test_preds, scaled_df, target):
    confusion_matrix(y_test, test_preds)
    classification_report(y_test, test_preds)
    print('Mean Adaboost Cross-Val Score (k=5):')
    print(cross_val_score(model, scaled_df, target, cv=5).mean())