import os
import pickle

import pandas as pd


class Reader:
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

    '''
    Read ADMISSIONS table (58.976)

    ROW_ID	        INT
    SUBJECT_ID	    INT
    HADM_ID	        INT
    ADMITTIME	    TIMESTAMP(0)
    DISCHTIME	    TIMESTAMP(0)
    DEATHTIME	    TIMESTAMP(0)
    ADMISSION_TYPE	VARCHAR(50)
    ADMISSION_LOCATION	VARCHAR(50)
    DISCHARGE_LOCATION	VARCHAR(50)
    INSURANCE	VARCHAR(255)
    LANGUAGE	VARCHAR(10)
    RELIGION	VARCHAR(50)
    MARITAL_STATUS	VARCHAR(50)
    ETHNICITY	VARCHAR(200)
    EDREGTIME	TIMESTAMP(0)
    EDOUTTIME	TIMESTAMP(0)
    DIAGNOSIS	VARCHAR(300)
    HOSPITAL_EXPIRE_FLAG	TINYINT
    HAS_CHARTEVENTS_DATA	TINYINT
    '''

    def read_admissions_table(self):
        df = pd.read_csv(os.path.join(self.path_dataset, 'ADMISSIONS.csv'), header=0, index_col=0)
        df.columns = map(str.upper, df.columns)
        df.ADMITTIME = pd.to_datetime(df.ADMITTIME)
        df.DISCHTIME = pd.to_datetime(df.DISCHTIME)
        df = df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME','ETHNICITY']]
        return df

    '''
    Read ICUSTAYS table (61,532)

    ROW_ID	    INT
    SUBJECT_ID	INT
    HADM_ID	    INT
    ICUSTAY_ID	INT
    DBSOURCE	VARCHAR(20)
    FIRST_CAREUNIT	VARCHAR(20)
    LAST_CAREUNIT	VARCHAR(20)
    FIRST_WARDID	SMALLINT
    LAST_WARDID	    SMALLINT
    INTIME	    TIMESTAMP(0)
    OUTTIME	    TIMESTAMP(0)
    LOS	        DOUBLE
    '''

    def read_icustay_table(self):
        df = pd.read_csv(os.path.join(self.path_dataset, 'ICUSTAYS.csv'), header=0, index_col=0)
        df.columns = map(str.upper, df.columns)
        df.INTIME = pd.to_datetime(df.INTIME)
        df.OUTTIME = pd.to_datetime(df.OUTTIME)
        df = df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']]
        return df

    '''
    Read D_ICD_DIAGNOSES table (14.567)

    ROW_ID	INT
    ICD9_CODE	VARCHAR(10)
    SHORT_TITLE	VARCHAR(50)
    LONG_TITLE	VARCHAR(300)

    '''

    def read_d_icd_diagnoses_table(self):
        d_icd_diagnoses = pd.read_csv(os.path.join(self.path_dataset, 'D_ICD_DIAGNOSES.csv'), header=0, index_col=0)
        d_icd_diagnoses.columns = map(str.upper, d_icd_diagnoses.columns)
        return d_icd_diagnoses

    '''
    Read D_ITEMS table (12.487)

    ROW_ID	INT
    ITEMID	INT
    LABEL	VARCHAR(200)
    ABBREVIATION	VARCHAR(100)
    DBSOURCE	VARCHAR(20)
    LINKSTO	VARCHAR(50)
    CATEGORY	VARCHAR(100)
    UNITNAME	VARCHAR(100)
    PARAM_TYPE	VARCHAR(30)
    CONCEPTID	INT
    '''

    def read_d_items_table(self):
        d_items = pd.read_csv(os.path.join(self.path_dataset, 'D_ITEMS.csv'), header=0, index_col=0)
        d_items.columns = map(str.upper, d_items.columns)
        d_items = d_items[['ITEMID', 'LABEL', 'DBSOURCE', 'PARAM_TYPE']]
        return d_items

    '''
    Read D_LABITEMS tables (753)

    ROW_ID	INT
    ITEMID	INT
    LABEL	VARCHAR(100)
    FLUID	VARCHAR(100)
    CATEGORY	VARCHAR(100)
    LOINC_CODE	VARCHAR(100)
    '''

    def read_d_labitems_table(self):
        d_labitems = pd.read_csv(os.path.join(self.path_dataset, 'D_LABITEMS.csv'), header=0, index_col=0)
        d_labitems.columns = map(str.upper, d_labitems.columns)
        d_labitems = d_labitems[['ITEMID', 'LABEL', 'FLUID', 'CATEGORY']]
        return d_labitems

    '''
    Read PATIENTS table (46.520)

    ROW_ID	INT
    SUBJECT_ID	INT
    GENDER	VARCHAR(5)
    DOB	TIMESTAMP(0)
    DOD	TIMESTAMP(0)
    DOD_HOSP	TIMESTAMP(0)
    DOD_SSN	TIMESTAMP(0)
    EXPIRE_FLAG	VARCHAR(5)

    '''
    
    def read_patients_table(self):
        patients = pd.read_csv(os.path.join(self.path_dataset, 'PATIENTS.csv'), header=0, index_col=0)
        patients.columns = map(str.upper, patients.columns)
        patients = patients[['SUBJECT_ID', 'GENDER', 'DOB']]
        return patients

    '''
     Read DIAGNOSES_ICD table (651.047)

    ROW_ID	INT	
    SUBJECT_ID	INT	
    HADM_ID	INT
    SEQ_NUM	INT	
    ICD9_CODE	VARCHAR(10)	

    '''

    def read_diagnoses_icd_table(self):
        diagnoses_icd = pd.read_csv(os.path.join(self.path_dataset, 'DIAGNOSES_ICD.csv'), header=0, index_col=0)
        diagnoses_icd.columns = map(str.upper, diagnoses_icd.columns)
        diagnoses_icd = diagnoses_icd[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']]
        return diagnoses_icd

    '''
    Load data train and test
    '''

    def read_data(self, name, dataset='timeseries_inter'):
        path_file = os.path.join('/home/quanglv/Early-Prediction-of-AKI-in-ICU/data/create_dataset/', dataset)
        with open(path_file + '/' + name + '.pickle', 'rb') as f:
            data = pickle.load(f)
            samples = data['samples']
            labels = data['labels']
            print(name.capitalize() + ' shape: {}'.format(samples.shape))
            print('Data ratio: {}/{}'.format(labels[labels == 1].shape[0], labels[labels == 0].shape[0]))
            return samples, labels
