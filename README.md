
# Deep Learning model for AKI Prediction on MIMIC-III

Aki-Predictor is a set of python script running deep learning model to predict Acute Kiden Injuries during the first 7 days of stay in ICU. The proposed model was tested on MIMIC-III database.

We are releasing this repository to make it possible to replicate our work, and in case it is useful for further work in this area. If you are using any part of this code repository that we have added to, we would appreciate if you cite our paper: (reference to follow after review)

This repository is released under an GNU Affero General Public License. 

# System Requirements
# Dependencies

This repository requires conda to manage and install the many Python package dependencies required to run the experiments. Installing the dependencies using the conda environment file in the repository is detailed below.

In theory this repo is (mostly) platform agnostic through its use of Python and conda. In practice, it has only been tested on a relatively recent Linux system (5.4.0-96-generic #109-Ubuntu SMP), using conda 4.12.0. There is no requirement for non-standard hardware.

# Installation
# Dataset

A small demo data set for testing the code is available here: https://doi.org/10.13026/C2HM2Q

The main MIMIC-3 dataset is accessible after passing the necessary data handling training. More information can be found here: https://mimic.mit.edu/docs/iii/

# Code and Data preparation

Establish an appropriate conda environment (and then activate it):

```
conda create --name some_name --file conda/my_conda_env.yml
conda activate some_name
```

Expected install time is < 10 minutes on a reasonably modern system.

# Running experiments

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

To run the machine learning model execute ``` python aki-ml.py ``` . The model is a set of multilayer perceptrons. We used hyperparameter tuning to find the best architecture (a 15-layer deep learning network). We followed a 80% training - 20% testing partition of data with five-fold cross validation. We developed our model based on 83 features referring to routinely collected clinical parameters extracted using the scripts descirbed above.

```
The scripts contains the following functions:

* run_aki_model: predicts wether a patient will develop AKI withnin the first 7 days of its stay and which stage of AKI it is according to the KIDIGO guidelines.
* cluster_ethnicity: subsets the data  by  ethnicity:  train  on  ”Caucasian”  (all variants),  predict  for  all  other  ethnicities.   
* change_data_size: does random subsampling of available training data
* cluster_drift: subset of the data by EMR system: train on pre-2008 data and test it on the subset post-2008, ignore patients that appear in both systems.
```
# Expected output

The main expected output of the postgres and the pre-process scripts are csv files with data extracted and cleaned from either the demo or the main dataset. The output of the aki-ml.py are performances in achieved AUROCs of the model on the test sets for each scenario. 
