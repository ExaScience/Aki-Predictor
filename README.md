# Deep Learning model for AKI Prediction on MIMIC-III

Aki-Predictor is a set of python script running deep learning model to predict Acute Kiden Injuries during the first 7 days of stay in ICU. The proposed model was tested on MIMIC-III database.

We developed our model based on 83 features referring to routinely collected clinical parameters.  
The features includes demographics data, vital signs measured at the bedsidesuch as heart rate, arterial blood pressure, respiration rate, etc. laboratory test results such 
as blood urea nitrogen, hemoglobin, white blood count, etc. average of urine output, theminimum  value  of  estimated  glomerular  filtration  rate  (eGFR)  and  creatinine.
We also included co-morbidities such as congestive heart failure,  hypertension,  diabetes,  etc.

Execute following script to extract AKI patient data from the MIMIC III tables.

```
python aki-postgres.py
```

Execute following script to to clean and preprocess the csv files generated from the data extraction step.

```
python aki-preprocess.py
```

To run the machine learning model 

```
python aki-ml.py
```

The scripts contains the following functions:

* run_aki_model: predicts wether a patient will develop AKI withnin the first 7 days of its stay and which stage of AKI it is according to the KIDIGO guidelines.
* cluster_ethnicity: subsets the data  by  ethnicity:  train  on  ”Caucasian”  (all variants),  predict  for  all  other  ethnicities.   
* change_data_size: does random subsampling of available training data

