import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def scale_train_test_split(X,y, set_seed=101):
    """
        Get predictors and target and return training and test splits
    """
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    target = y # Need to modify depending on target column
    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.25, random_state = set_seed, stratify=y)
    return X_train, X_test, y_train, y_test

def train_model(X_train,y_train):
    """ 
        Fit several models
    """
    model_names = []
    models = []
    # Adaboost
    
    model_names.append('AdaBoost')
    print('Running ' + model_names[-1])
    adaboost_clf = AdaBoostClassifier()
    adaboost_clf.fit(X_train, y_train)
    models.append(adaboost_clf)
    
    # GradientBoost
#     model_names.append('GradientBoost')
#     print('Running ' + model_names[-1])
#     gbt_clf = GradientBoostingClassifier()
#     gbt_clf.fit(X_train, y_train)
#     models.append(gbt_clf)
    
    # # XGBoost
#     model_names.append('XGBoost')
#     print('Running ' + model_names[-1])
#     xgb_clf = xgb.XGBClassifier()
#     xgb_clf.fit(X_train, y_train)
#     models.append(xgb_clf)
    
    # Logistic Regression
    model_names.append('Logistic Regresion')
    print('Running ' + model_names[-1])
    lr_clf = LogisticRegression(fit_intercept = False, C = 1e12)
    lr_clf.fit(X_train, y_train)
    models.append(lr_clf)
    
    # Decision Tree
    model_names.append('DecisionTree_gini')
    print('Running ' + model_names[-1])
    dtree_clf = DecisionTreeClassifier(criterion='gini', max_depth=8)
    dtree_clf.fit(X_train, y_train)
    models.append(dtree_clf)

#     model_names.append('DecisionTree_entropy')
#     print('Running ' + model_names[-1])
#     dtree_clf2 = DecisionTreeClassifier(criterion='entropy', max_depth=8)
#     dtree_clf2.fit(X_train, y_train)
#     models.append(dtree_clf2)
    
    return models, model_names

def predict_all(x1, y1, models):
    """
    Return the predictions in a list, one element per model passed in models
    Input:
    x1: Features
    y1: Targets
    models: list of models
    """
    predictions = []
    for model in models:  
        predictions.append(model.predict(x1))
    
    return predictions
    
def display_acc_and_f1_score(true, preds, model_name):
    """
    Calculate the accuracy and the f1 scores 
    """
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
    
def find_best_tree_clf(X_train,y_train):
    dtree_clf = DecisionTreeClassifier()
    pipe = Pipeline([('clf', dtree_clf)])
    param_grid = [{'clf__criterion':['gini','entropy'],
                  'clf__max_depth':[2,3,5,10,15],
                  'clf__min_samples_split':[2,5,6]}]
    grid = GridSearchCV(pipe, param_grid=param_grid, cv= 5, verbose=1,n_jobs=2)
    
    grid.fit(X_train, y_train)
    return grid

def find_best_logistic(X_train, y_train):
    logistic = LogisticRegression(fit_intercept = False)
    pipe = Pipeline([('logistic',logistic)])
    C = np.logspace(-4,4,5)
    penalty = ['l1','l2']
    
    parameters = dict(logistic__C = C, 
                     logistic__penalty=penalty)
    grid = GridSearchCV(pipe, param_grid=parameters, cv=5, verbose=1, n_jobs=2)
    
    grid.fit(X_train, y_train)
    return grid