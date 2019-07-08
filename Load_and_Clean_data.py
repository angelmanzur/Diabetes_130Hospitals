import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

def load_raw_data():
    filename = 'dataset_diabetes/diabetic_data.csv'
    df = pd.read_csv(filename)
    print('Loaded data into dataframe')
    return df

def clean_data(df):
    # drop some unused columns
    df.drop(columns = ['patient_nbr','citoglipton','weight','examide','encounter_id'],inplace=True)
    # weight, drop coz 97% missing
    # citoglipton, and examide are constant values
    # encounter_id is just an id

    # race, missing set to other 2%
    df.race.replace('?', 'Other',inplace=True)  

    df.drop(columns = ['diag_1', 'diag_2', 'diag_3', 'payer_code'], inplace=True)

    # convert change from No, Ch to 0, 1
    df.change = np.where(df.change=='No',0,1)
#     df.medication_change = np.where(df.change=='No',0,1)
#     df.drop('change', axis=1, inplace=True)
    
    # convert diabetes_med from Yes or No to 1 or 0
    df.diabetesMed = np.where(df.diabetesMed=='Yes',1,0)

    df.drop('medical_specialty', axis=1, inplace=True)
    # 49,000 are ?

    df = df[(df.gender=='Male') | (df.gender=='Female')]
    df.readmitted = np.where(df.readmitted=='NO', 0, df.readmitted )
    df.readmitted = np.where(df.readmitted=='<30', 1,  df.readmitted )
    df.readmitted = np.where(df.readmitted=='>30', 2,  df.readmitted )

    
    return df

def clean_eng_data(df):
    # drop some unused columns
    df.drop(columns = ['patient_nbr','citoglipton','weight','examide','encounter_id'],inplace=True)
    # weight, drop coz 97% missing
    # citoglipton, and examide are constant values
    # encounter_id is just an id

    # race, missing set to other 2%
    df.race.replace('?', 'Other',inplace=True)  

#     df.drop(columns = ['diag_1', 'diag_2', 'diag_3', 'payer_code'], inplace=True)
    df.drop(columns = ['payer_code'], inplace=True)

    # convert change from No, Ch to 0, 1
    df.change = np.where(df.change=='No',0,1)
#     df.medication_change = np.where(df.change=='No',0,1)
#     df.drop('change', axis=1, inplace=True)
    
    # convert diabetes_med from Yes or No to 1 or 0
    df.diabetesMed = np.where(df.diabetesMed=='Yes',1,0)

    df.drop('medical_specialty', axis=1, inplace=True)
    # 49,000 are ?
    
    df = df[(df.gender=='Male') | (df.gender=='Female')]
    df.readmitted = np.where(df.readmitted=='NO', 0, df.readmitted )
    df.readmitted = np.where(df.readmitted=='<30', 1,  df.readmitted )
    df.readmitted = np.where(df.readmitted=='>30', 2,  df.readmitted )
    df['readmitted'] = df['readmitted'].astype(int)
    df.drop(columns=[ 'admission_source_id'], inplace=True)
    
    return df
 
def df_featues_target(df):


    dummy_df = pd.get_dummies(df,drop_first=True)
    to_drop = ['readmitted_1', 'readmitted_2']
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

def count_meds(df, meds_cols):
    #  Convert the medicines based on
    #  Steady or No = 0
    #  Up -> 1
    #  Down -> -1


    for med in meds_cols:
        df[med] = np.where((df[med]=='No' ),0,df[med])
        df[med] = np.where((df[med]=='Steady' ),0,df[med])
        factor = 1
        if(med=='insulin'):
            factor=2

        df[med] = np.where((df[med]=='Up' ), 1*factor,df[med])
        df[med] = np.where((df[med]=='Down' ), -1*factor,df[med])

    df['dosage_up_down'] = 0
#     df['dosage_up'] = 0
#     df['dosage_down'] = 0
    for med in meds_cols:
        df['dosage_up_down'] += df[med]
#         df['dosage_up'] += np.where(df[med]>0, df[med], 0)
#         df['dosage_down'] += np.where(df[med]<0, df[med], 0)
    
        
    # remove the medicines
    df.drop(meds_cols, axis=1, inplace=True)
    

        
    return df

def get_diagnosis(df, col_name):
    print('converting diagnosis for', col_name)
    new_name = 'cat1_'+col_name
    df[new_name] = df[col_name]
    
    df[new_name] = np.where( ((df[col_name].str.contains('V', na=False)) | 
                            (df[col_name].str.contains('E', na=False)) ), 9, df[new_name])
    df[new_name] = np.where(df[new_name]=='?', -1, df[new_name] )
    
    df[new_name] = df[new_name].astype(float)
    
    df[new_name] = np.where( (((df[new_name]>=390) & (df[new_name]<460)) | (np.floor(df[new_name])==785) ),1, df[new_name] )
    
    df[new_name] = np.where( (((df[new_name]>=460) & (df[new_name]<520)) | (np.floor(df[new_name])==786) ),2, df[new_name] )
    
    df[new_name] = np.where( (((df[new_name]>=520) & (df[new_name]<580)) | (np.floor(df[new_name])==787) ),3, df[new_name] )
    
    df[new_name] = np.where( (np.floor(df[new_name])==250  ) ,4,df[new_name])
    
    df[new_name] = np.where( ((df[new_name] >=800) & (df[new_name]<1000)),5,df[new_name])
    
    df[new_name] = np.where( ((df[new_name] >=710) & (df[new_name]<740)),6,df[new_name])
    
    df[new_name] = np.where( (((df[new_name]>=580) & (df[new_name]<630)) | (np.floor(df[new_name])==788) ),7, df[new_name] )
    
    df[new_name] = np.where( ((df[new_name] >=140) & (df[new_name]<240)),8,df[new_name])
    
    df[new_name] = np.where( df[new_name] >9, 9, df[new_name])
    
    return df

def get_visits(df):
    df['total_visits'] = df.number_inpatient + df.number_outpatient + df.number_emergency
    # df.drop(columns=['number_inpatient', 'number_outpatient','number_emergency'],inplace=True)
    df.drop(columns=['number_inpatient', 'number_outpatient'],inplace=True)
    return df

def get_admissions(df):
    
    df['admission'] = 0
    df['admission'] = np.where( df.admission_type_id <3 , 1, 0 )
    df['admission'] = np.where( df.admission_type_id == 7, 1, df.admission )
    df.drop(columns=['admission_type_id'], inplace=True)
    
    return df


def get_procedures(df):
    df.total_procedures = df.num_lab_procedures +    df.num_procedures
    df.drop(columns=['num_lab_procedures', 'num_procedures'], inplace=True)

    return df

def get_LAE_index(df):
    # Creating an index based on the LACE index
    # L day in the hospital
    # A : acuity of admission,  3 for emergency, 0 otherwise
    # E : number of emergency visits 4 or smaller
    
    df['LAE_index'] = df['time_in_hospital']
# get the L value

    df['LAE_index'] = np.where(((df['time_in_hospital']>4) & (df['time_in_hospital']<7)), 4, df['LAE_index'])
    df['LAE_index'] = np.where(((df['time_in_hospital']>6) & (df['time_in_hospital']<14)), 5, df['LAE_index'])
    df['LAE_index'] = np.where( (df['time_in_hospital']>13), 7, df['LAE_index'])
    
    df['LAE_index'] +=  df['admission']*3
    
    df['LAE_index'] = np.where( df['number_emergency']>=4, 
                           df['LAE_index'] +4,    df['LAE_index']+df['number_emergency'])
    
    return df

def get_LAMA_index(df):
    df['LAMA'] = 0
    df['LAMA'] = np.where( df['discharge_disposition_id']==7, 1, 0)
    return df

def get_glu_serum(df):
    df['max_glu_serum'] = np.where(df['max_glu_serum']=='>300', 1,df['max_glu_serum'])
    df['max_glu_serum'] = np.where(df['max_glu_serum']=='>200', 1,df['max_glu_serum'])
    df['max_glu_serum'] = np.where(df['max_glu_serum']=='Norm', 0,df['max_glu_serum'])
    df['max_glu_serum'] = np.where(df['max_glu_serum']=='None', 0,df['max_glu_serum'])
    df['max_glu_serum'] = df['max_glu_serum'].astype(int)
    return df

def get_A1Cresults(df):
    df['A1Cresult'] = np.where(df['A1Cresult']=='>8', 1,df['A1Cresult'])
    df['A1Cresult'] = np.where(df['A1Cresult']=='>7', 1,df['A1Cresult'])
    df['A1Cresult'] = np.where(df['A1Cresult']=='Norm', 0,df['A1Cresult'])
    df['A1Cresult'] = np.where(df['A1Cresult']=='None', 0,df['A1Cresult'])
    df['A1Cresult'] = df['A1Cresult'].astype(int)
    
    return df

def get_age_bin(dummy_df):
    dummy_df['age_60_100'] = dummy_df['age_[60-70)']+dummy_df['age_[70-80)']+dummy_df['age_[80-90)']+dummy_df['age_[90-100)']
    dummy_df['age_30_60'] = dummy_df['age_[30-40)']+dummy_df['age_[40-50)']+dummy_df['age_[50-60)']
    dummy_df['age_0_30'] = dummy_df['age_[0-10)']+dummy_df['age_[10-20)']+dummy_df['age_[20-30)']

    to_drop = ['age_[0-10)', 'age_[10-20)', 'age_[20-30)', 'age_[30-40)',
       'age_[40-50)', 'age_[50-60)', 'age_[60-70)', 'age_[70-80)',
       'age_[80-90)', 'age_[90-100)', 'gender_Male', 'race_Other']
    dummy_df.drop(columns=to_drop,inplace=True)
    return dummy_df

def get_one_diagnosis(cleaned_df):
    cleaned_df['diagnosis'] = cleaned_df['cat1_diag_1']
    cleaned_df['diagnosis'] = np.where( ((cleaned_df['cat1_diag_2']==cleaned_df['cat1_diag_3']) & (cleaned_df['cat1_diag_2']>0)),
                    cleaned_df['cat1_diag_2'], cleaned_df['cat1_diag_1'] )
    cleaned_df['diagnosis'] = np.where( cleaned_df['diagnosis']==-1, cleaned_df['cat1_diag_2'], cleaned_df['diagnosis'])
    cleaned_df['diagnosis'] = cleaned_df['diagnosis'].astype(int)
    cleaned_df['diagnosis'] = cleaned_df['diagnosis'].astype(str)
    diagnosis_df = pd.get_dummies(cleaned_df)
    diagnosis_df.drop(columns=['cat1_diag_1', 'cat1_diag_2','cat1_diag_3'], inplace=True)
    return diagnosis_df

def remove_patients(diagnosis_df):
    print('number of patients', len(diagnosis_df))
    diagnosis_df = diagnosis_df[diagnosis_df['diagnosis_-1']==0]
    diagnosis_df.drop(columns=['diagnosis_-1'], inplace=True)
    len(diagnosis_df[diagnosis_df.discharge_disposition_id==11])
    diagnosis_df = diagnosis_df[diagnosis_df.discharge_disposition_id!=11]
    # remove patients that have moved to hospice (home or medical facility)
    diagnosis_df = diagnosis_df[diagnosis_df.discharge_disposition_id!=13]
    diagnosis_df = diagnosis_df[diagnosis_df.discharge_disposition_id!=14]

    # remove patients expired at home. medical facility or unknown
    diagnosis_df = diagnosis_df[diagnosis_df.discharge_disposition_id!=19]
    diagnosis_df = diagnosis_df[diagnosis_df.discharge_disposition_id!=20]
    diagnosis_df = diagnosis_df[diagnosis_df.discharge_disposition_id!=21]
    print('number of patients', len(diagnosis_df))
    
    return diagnosis_df
def sort_asc_corr(df, value):
    ''' Return correlation in descending order
        based on the specified value parameter
        in our case, we will be using the
        dependent variable price '''

    corr_df = df.corr()[[value]]
    return corr_df.sort_values(by = value, ascending=False)

def save_X_y(X,y):
    X.to_csv('dataset_diabetes/features.csv', index=False)
    y.to_csv('dataset_diabetes/target.csv',index=False)

    pass

def load_X_y():
    X = pd.read_csv('dataset_diabetes/features.csv')
    y = pd.read_csv('dataset_diabetes/target.csv',header=None)
    y.columns = ['readmitted']
    return X,y['readmitted']



from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=None):
    #=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax