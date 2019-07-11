# Predicting hospital readmission for diabetic patients

# Objective

# Motivation

# Data
The data was downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008). The dataset consists of +100,000 patients admitted into a hospital and diagnosed with diabetes. The data was compiled from 130 US hospitals between 1998 and 2008.

The patients included in the dataset satisfy the following criteria:
    - The patients were diagnosed as diabetic.
    - The patiends stained in the hospital between 1 and 14 days. 
    - Laboratory tests were performed during the encounter.
    - Medications were administered during the encounter

The data contains 55 features including:
  - Race of the patient, one out of five categorical values.
  - Gender
  - Age, in 10 year bins.
  - Weight in pounds.
  - Admission type, for example emergency, urgent or new born. One out of nine possible values.
  - Discharge type, for example disharged to home or expired. One out of 29 possible values
  - Admission source, for example physician referral or emergency room. One out of 21 possible values
  - Time in the hospital, in days
  - Payer code, one of 23 possible values.
  - Medical specialty od the admitting physician, one out of 84 distinct values.
  - Number of lab procedures performed.
  - Numper of procedures (other than lab tests) performed.
  - Number of medications administered during the encounter.
  - Number of outpatient and inpatient in visits in the year preceding the encounter.
  - Number of emergency visits in the year preceding the encounter.
  - The three primary diagnosis codes (if applicable), more than 900 distinct values.
  - Number of diagnosis entered into the system.
  - Glucose serum test result: >200, >300, normal or none.
  - A1C test result: >7%, >8%, normal or none.
  - Information on 23 diabetic medications: the data shows if the dosage change or not. medicines
  - Change on the diabetics medications.
  - Readmitted. Information if the patient was readmitted in more than 30 days, less than 30 days or if they were not readmitted.
For a full description of the features see references at the end of this page. 


# Base Model
Our first model consisted on minimal clenaing of the data:
    - Removed variables that had unique values, such as patient ID and  encounter ID, variables that were the same for all patiends, like 2 of the medicines
    - Removed columns with a significant fraction of missing data: 97% of the data were missing weight, 49% were missing medical specialty.
    - Dropped the diagnosis, as we had +900 unique codes.
    - Made the race, gender and medicines, admission and discharges types into categorical data.
    
As an initial test we split the data into a train and test datasets, and ran a an AdaBoost, a Logistic Regression, and a Decision Tree model. The figure below shows the confusion matrix after fitting a Logistic Classifier on the training data and running it on the testing data. The accuracy of this model was 0.512.
    
![Confusion matrix for a Logistic Regression Classifier](Figures/Base_Model/Test_Logistic_Regresion.png)

The rest of the methods tested gave similar results. In our previous experience fine tunning the model parameters would ounly increase the accuracy by a few percent, instead of investigating different models and their parameter space, we decided to do some feature engineering to improve the resutls.

# Data Engineering
The majority of the time spent on this project was on understanding the data and trying to find  the relevant features. With the 550+ features the data set has, ther are thousands of possible feature combinations. Instead of testing which ones contribute, and which ones don't we engineered our features based on our knowledge on the topic. 

For our final model, we treated the dadta in the following way:
 - Removed features that had a constant value on all patients, removed unique IDs, and removed variables with values missing on a significant fraciont of tha patients, see previous section. 
 - The target variable was changed to predict onlt patiends readmitted in less than 30 dys. the target variable change according to the table below
 
 | Original Target Value | New Target Value  |
 | :---------------:|  :--------------:|
 | Not readmitted  |   0   |
 |  <30  |  1  |
 |  >30  |  0  |

- The 23 medicines dosage were summariazed into one variable, `dosage_up _down`. If the dosage of any medicine increased or decreased, we added or substacted a 1 the the variable. 
 - The 900+ diagnosis categories were binned into 9 different: Circuilatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasma, or Other. We kept the primary diagnosis given, unless the second  diagnosis belonged to the same categoryh as the third diagnosis. In those cases we kept the second diagnosis. 
 - Summed all the visits in the preceding year into a single feature.
 - Summed the lab and non-lab procedures into a single feature.
 - Create a new LAE index, defined as `L + A + E`, where: 
      - `L` if the length of stay in the hospital
      - `A` is the acuity of the admission, `A=3` for emergencies, and 0 otherwise
      - `E` is the number of emergency visits in the past year, maxed out at 4. 
 - The results of the glucose and the A1C   tests were set to according to the table below:
 | Original Glucose Serum | New Glucose Serum || Original A1C results | New A1C results |
 | :--: | :---: | :-: | :---: | :---: |
 | >300 | 3 | | >8 | 10 |
 | >200 | 1 | | >7 | 1 |
 | Norm | 0 | | Norm | 0 |
 | None | 0 | | None | 0 |

 - The age categories were grouped into three age groups: from 0 to 30 years old,  30 to 60 years old, and 60 to 100 years old.
 - We also added a new columns identifying if a patient left the hospital against medical advice (LAMA).

After selecting these features, we dropped the patiends that would not return beacause they died in the hospital or were sent to a hospice. This reduced our data set from 101763 patients down to 99340 patients. Out of those 99340, only 11314 patients were readmitted within 30 days. Sincce our data set is imbalanced, we use the stratify optin in the train_test_split method in SciKit-Learn, to force the train and test data sets to have the same ratio of readmitted patiets. 

```python
from sklearn.model_selectrion import train_test_split
X_train, X_test, y_trian, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=101
```



# Results

# Summary

# References

[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

[Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. Strack et. al. BioMed Research International, Wol 2014, Art. ID 781679](http://dx.doi.org/10.1155/2014/781670)

