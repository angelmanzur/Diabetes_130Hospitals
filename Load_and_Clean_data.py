import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

def load_raw_data():
    filename = 'dataset_diabetes/diabetic_data.csv'
    df = pd.read_csv(filename)
    return df

def clean_data(df):
    # drop some unused columns
    df.drop(columns = ['patient_nbr','citoglipton','weight','examide','encounter_id', 'payer_code'],inplace=True)
    # weight, drop coz 97% missing
    # citoglipton, and examide are constant values
    # encounter_id is just an id

    # race, missing set to other 2%
    df.race.replace('?', 'Other',inplace=True)  

#     df.drop(columns = ['diag_1', 'diag_2','diag_3'],inplace=True)

    # convert change from No, Ch to 0, 1
    df.medication_change = np.where(df.change=='No',0,1)
    df.drop('change', axis=1, inplace=True)
    
    # convert diabetes_med from Yes or No to 1 or 0
    df.diabetesMed = np.where(df.diabetesMed=='Yes',1,0)

    df.drop('medical_specialty', axis=1, inplace=True)
    # 49,000 are ?

    df.readmitted = np.where(df.readmitted=='NO', 0, df.readmitted )
    df.readmitted = np.where(df.readmitted=='<30', 1,  df.readmitted )
    df.readmitted = np.where(df.readmitted=='>30', 0,  df.readmitted )

    return df

    # set the target to 

def df_featues_target(df):


    dummy_df = pd.get_dummies(df,drop_first=True)
    to_drop = ['readmitted_1', 'readmitted_2','gender_Unknown/Invalid']
    for cname in to_drop:
        try:
            dummy_df.drop(cname, axis=1,inplace=True)
        except:
            pass


    X = dummy_df.copy()
    col_names = X.columns
    col_names_list = []
    for i, col_name in enumerate(col_names):
        c1 = col_name.replace('[','')
        c1 = c1.replace('>','')
        col_names_list.append(c1)
    
    X.columns = col_names_list
    y = df.readmitted.astype(int)
  
    return X, y

def save_X_y(X,y):
    X.to_csv('dataset_diabetes/features.csv', index=False)
    y.to_csv('dataset_diabetes/target.csv',index=False)

    pass

def load_X_y():
    X = pd.read_csv('dataset_diabetes/features.csv')
    y = pd.read_csv('dataset_diabetes/target.csv',header=None)
    y.columns = ['readmitted']
    return X,y['readmitted']