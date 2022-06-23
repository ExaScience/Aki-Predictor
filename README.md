# Deep Learning model for AKI Prediction on MIMIC-III

Aki-Predictor is a set of python script running deep learning model to predict Acute Kiden Injuries during the first 7 days of stay in ICU. The proposed model was tested on MIMIC-III database.

We developed our model based on 83 features referring to routinely collected clinical parameters.  
The features includes demographics data, vital signs measured at the bedsidesuch as heart rate, arterial blood pressure, respiration rate, etc. laboratory test results such  as blood urea nitrogen, hemoglobin, white blood count, etc. average of urine output, theminimum  value  of  estimated  glomerular  filtration  rate  (eGFR)  and  creatinine.

We also included co-morbidities such as congestive heart failure,  hypertension,  diabetes,  etc.

Execute following script ``` python aki-postgres.py ``` to extract AKI patient data from the MIMIC III tables.

```
The scripts contains the following functions:

* get_comorbidities: This query extracts list of comorbidities of the patients upon admission in the ICU.

* get_vitals_chart: This query extracts the vital signs during the first 7 days of a patient's stay

* get_labevents: This query extracts lab values taken during the 7 first days of a patient's stay

* kidigo_7_days_creatinine: This query checks if the patient had AKI during the first 7 days of their ICU stay according to the KDIGO guideline and based on only creatinine feature.

* kidigo_7_days: This query checks if the patient had AKI during the first 7 days of their ICU stay according to the KDIGO guideline and based on creatinine and urine feature.

```
Execute following script ``` python aki-preprocess.py ``` to clean and preprocess the csv files generated from the data extraction step.

```
The scripts contains the following functions:

* check_AKI_before: checks if a patient has AKI before admission to the ICU
* get_info_admissions: selects informations about the admission
* check_CKD: checks if the patient has chronic kidney disease before admission to the ICU
* calculate_eGFR_MDRD_equation: calculates the the minimum value of estimated glomerular filtration rate

```

To run the machine learning model execute ``` python aki-ml.py ``` . The model is a set of multilayer perceptrons. We used hyperparameter tuning to find the best architecture (a 15-layer deep learning network). We followed a 80% training - 20% testing partition of data with five-fold cross validation.

```
The scripts contains the following functions:

* run_aki_model: predicts wether a patient will develop AKI withnin the first 7 days of its stay and which stage of AKI it is according to the KIDIGO guidelines.
* cluster_ethnicity: subsets the data  by  ethnicity:  train  on  ”Caucasian”  (all variants),  predict  for  all  other  ethnicities.   
* change_data_size: does random subsampling of available training data
* cluster_drift: subset of the data by EMR system: train on pre-2008 data and test it on the subset post-2008, ignore patients that appear in both systems.
```
