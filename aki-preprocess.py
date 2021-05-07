import os
import sys

import numpy as np
import pandas as pd

import os
from chunk import Chunk

import time

from util.reader import Reader
from util.util import create_folder

path_preprocess_data = './preprocessing'

create_folder(path_preprocess_data)

path_dataset = 'mimic-iii-clinical-database-1.4' #Path to yoyr mimic database 

reader = Reader(path_dataset=path_dataset)

def get_info_admissions():
   
    df = reader.read_admissions_table()
    df['STAYTIME'] = df['DISCHTIME'] - df['ADMITTIME'] #staty time : discharge time - admission time 
    df['STAYTIME'] = df['STAYTIME'] / np.timedelta64(1, 'h')
  
    #formula to calcultate the age of patiens in MIMIC3 
    
    patients = reader.read_patients_table()
    df = pd.merge(df, patients, how='left', on='SUBJECT_ID')
    df['DOB'] = pd.to_datetime(df['DOB'])
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['AGE'] = (df['ADMITTIME'].dt.year - df['DOB'].dt.year)

    #Patients who are older than 89 years old at any time in the database 
    #have had their date of birth shifted to obscure their age and comply with HIPAA. 
    #The date of birth was then set to exactly 300 years before their first admission.
    df.loc[((df.AGE > 89) | (df.AGE < 0)), 'AGE'] = 90

    #select patiens older than 18 
  
    icustays = reader.read_icustay_table()
  
    #merge on the HADM_ID, unique, represents a single patient's admission to the hospital
    #while subject_id can be redundant meaning that a patient had many stays at the hospital
    df = pd.merge(df, icustays, how='right', on='HADM_ID')

    #the elapsed time between the admission in the hospital and the tranfer to the ICU
    df['Time go ICU'] = (df['INTIME'] - df['ADMITTIME']) / np.timedelta64(1, 'h')

    #the elapsed time in the ICU
    df['Time in ICU'] = (df['OUTTIME'] - df['INTIME']) / np.timedelta64(1, 'h')

    # the elapsed time between the admission in the ICU and the final discharge from the hospital
    df['Time after go ICU'] = (df['DISCHTIME'] - df['INTIME']) / np.timedelta64(1, 'h')

    #how many time the patient has been transferrred to the ICU during one admission
    df['Count times go ICU'] = df.groupby('HADM_ID')['ICUSTAY_ID'].transform('count')
    
    if os.path.exists("demofile.txt"):
      os.remove("demofile.txt")
    else:
      print("The file does not exist")
  
    with open(os.path.join(path_preprocess_data,'ADMISSIONS.csv'), 'w') as f:
        df.to_csv(f, encoding='utf-8', header=True)
         
def check_AKI_before(hadm_id):
    
    diagnoses = pd.read_csv(os.path.join(path_dataset,'DIAGNOSES_ICD.csv'))
    diagnoses.columns = map(str.upper, diagnoses.columns)
    diagnoses = diagnoses.loc[diagnoses['ICD9_CODE'].isin(['5845', '5846', '5847', '5848'])]

    if not diagnoses[diagnoses['HADM_ID'].isin(hadm_id)].empty:
        return True
    
    return False

def check_CKD(hadm_id):
    
    diagnoses = pd.read_csv(os.path.join(path_dataset,'DIAGNOSES_ICD.csv'))
    diagnoses.columns = map(str.upper, diagnoses.columns)
    diagnoses = diagnoses.loc[diagnoses['ICD9_CODE'].isin(['5851', '5852', '5853', '5854', '5855'])]
   # print(diagnoses['HADM_ID'] , diagnoses[diagnoses['HADM_ID'].isin(hadm_id)])
    if not diagnoses[diagnoses['HADM_ID'].isin(hadm_id)].empty:
        return True
    
    return False

def check_renal_failure(hadm_id):
    
    diagnoses = pd.read_csv(os.path.join(path_preprocess_data,'comorbidities.csv'))
    diagnoses.columns = map(str.upper, diagnoses.columns)
    diagnoses = diagnoses.loc[diagnoses['RENAL_FAILURE'] == 1]
   # print(diagnoses['HADM_ID'] , diagnoses[diagnoses['HADM_ID'].isin(hadm_id)])
    if not diagnoses[diagnoses['HADM_ID'].isin(hadm_id)].empty:
        return True
    
    return False
    
def caculate_eGFR_MDRD_equation(cr, gender, eth, age):
    temp = 186 * (cr ** (-1.154)) * (age ** (-0.203))
    if (gender == 'F'):
        temp = temp * 0.742
    if eth == 'BLACK/AFRICAN AMERICAN':
        temp = temp * 1.21
    return temp

def get_aki_patients_7days():
     
    df = pd.read_csv(os.path.join(path_preprocess_data,'ADMISSIONS.csv'))
    df = df.sort_values(by=['SUBJECT_ID_x', 'HADM_ID', 'ICUSTAY_ID'])

    print("admissions info", df.shape)
    print("number of unique subjects in admission: " , df['SUBJECT_ID_x'].nunique())
    print("number of icustays info in admissions: ", df['ICUSTAY_ID'].nunique())
    
    info_save = df.drop_duplicates(subset=['ICUSTAY_ID'])
    info_save['AKI'] = -1
    info_save['EGFR'] = -1
        
    print("the biggest number of ICU stays for a patient: ", info_save['Count times go ICU'].max())
    
    c_aki_7d = pd.read_csv(os.path.join(path_preprocess_data,'AKI_KIDIGO_7D_SQL.csv'))
    c_aki_7d.columns = map(str.upper, c_aki_7d.columns)
    
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    print('NORMAL Patients in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1 within 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 1]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 2 in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 2]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 3 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 3]['ICUSTAY_ID'].count()))
    print('NAN patients within 7DAY: {}'.format(c_aki_7d['AKI_STAGE_7DAY'].isna().sum()))
    c_aki_7d = c_aki_7d.dropna(subset=['AKI_STAGE_7DAY'])
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    
    df_save = pd.merge(info_save, c_aki_7d, how='inner', on='ICUSTAY_ID')
    df_save.columns = map(str.upper, df_save.columns)
    icustays_data = [frame for season, frame in df_save.groupby(['ICUSTAY_ID'])]
    
    count_ckd_normal = 0
    count_ckd_aki = 0
    count_akibefore_normal = 0
    count_akibefore_aki = 0
    count_normal = 0
    count_aki = 0
    count_renalfailure_normal = 0
    count_renalfailure_aki = 0  
       
    for temp in icustays_data:
        
        temp = temp.sort_values(by=['ICUSTAY_ID'])
        
        gender = temp['GENDER'].values[0]
        age = temp['AGE'].values[0]
        eth = temp['ETHNICITY'].values[0]
        cr = temp['CREAT'].values[0]
         
        eGFR = caculate_eGFR_MDRD_equation(cr=cr, gender=gender, age=age, eth=eth)
                
        df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'EGFR'] = eGFR
        df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = c_aki_7d.loc[c_aki_7d['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0])]['AKI_7DAY'].values[0]
      
        if (df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
            count_aki = count_aki +1
        else:
            count_normal = count_normal + 1
              
        if (check_CKD(temp['HADM_ID']) == True):
            df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = 2
            if (info_save.loc[info_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
                count_ckd_aki = count_ckd_aki + 1
            else:
                count_ckd_normal = count_ckd_normal + 1
        
        if (check_AKI_before(temp['HADM_ID']) == True):
            df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = 3
            if (info_save.loc[info_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
                count_akibefore_aki = count_akibefore_aki + 1
            else:
                count_akibefore_normal = count_akibefore_normal + 1

        if (check_renal_failure(temp['HADM_ID']) == True):
            df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = 4
            if (info_save.loc[info_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
                count_renalfailure_aki = count_renalfailure_aki + 1
            else:
                count_renalfailure_normal = count_renalfailure_normal + 1

    lab = pd.read_csv(os.path.join(path_preprocess_data,'labstay.csv'))
    lab.columns = map(str.upper, lab.columns)
    info_save = pd.merge(df_save, lab, how='left', on='ICUSTAY_ID')
    info_save = info_save.drop(columns=['UNNAMED: 0_x','UNNAMED: 0_y'])
    info_save = info_save.rename(columns={'SUBJECT_ID_X': 'SUBJECT_ID', 'HADM_ID_x':'HADM_ID'})
  
    chart = pd.read_csv(os.path.join(path_preprocess_data,'chart_vitals_stay.csv'))
    chart.columns = map(str.upper, chart.columns)
    df_save = pd.merge(info_save, chart, how='left', on='ICUSTAY_ID')
    df_save = df_save.drop(columns=['UNNAMED: 0', 'HADM_ID_y', 'HADM_ID_y', 'SUBJECT_ID_Y', 'SUBJECT_ID_y'])
    df_save = df_save.rename(columns={'SUBJECT_ID_X': 'SUBJECT_ID', 'HADM_ID_x':'HADM_ID'})
   
    comorbidities = pd.read_csv(os.path.join(path_preprocess_data,'comorbidities.csv'))
    comorbidities.columns = map(str.upper, comorbidities.columns)
   
    info_save = pd.merge(df_save, comorbidities, how='left', on='HADM_ID')
    info_save = info_save.drop(columns=['UNNAMED: 0'])
      
    print('NORMAL Patients in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1 within 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 1]['ICUSTAY_ID'].count()))  
    print('CKD counted as normal: {}'.format(count_ckd_normal))
    print('CKD counted as aki: {}'.format(count_ckd_aki))
    print('AKI on admission counted as normal: {}'.format(count_akibefore_normal))
    print('AKI on admission counted as aki: {}'.format(count_akibefore_aki))
    print('RENAL FAILURE counted as normal: {}'.format(count_renalfailure_normal))
    print('RENAL FAILURE counted as aki: {}'.format(count_renalfailure_aki))
    print('normal: {}'.format(count_normal))
    print('aki: {}'.format(count_aki))
    
    with open(os.path.join(path_preprocess_data,'INFO_DATASET_7days_creatinine+urine2.csv'), 'w') as f:
        info_save.to_csv(f, encoding='utf-8', header=True)
   
def get_aki_patients_7days_creatinine():
     
    df = pd.read_csv(os.path.join(path_preprocess_data,'ADMISSIONS.csv'))
    df = df.sort_values(by=['SUBJECT_ID_x', 'HADM_ID', 'ICUSTAY_ID'])

    print("admissions info", df.shape)
    print("number of unique subjects in admission: " , df['SUBJECT_ID_x'].nunique())
    print("number of icustays info in admissions: ", df['ICUSTAY_ID'].nunique())
    
    info_save = df.drop_duplicates(subset=['ICUSTAY_ID'])
    info_save['AKI'] = -1
    info_save['EGFR'] = -1
        
    print("the biggest number of ICU stays for a patient: ", info_save['Count times go ICU'].max())
    
    c_aki_7d = pd.read_csv(os.path.join(path_preprocess_data,'AKI_KIDIGO_7D_SQL_CREATININE.csv'))
    c_aki_7d.columns = map(str.upper, c_aki_7d.columns)
    print("c_aki_7d infos")
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    print('NORMAL Patients in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1 within 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 1]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 2 in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 2]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 3 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 3]['ICUSTAY_ID'].count()))
    print('NAN patients within 7DAY: {}'.format(c_aki_7d['AKI_STAGE_7DAY'].isna().sum()))
    c_aki_7d = c_aki_7d.dropna(subset=['AKI_STAGE_7DAY'])
    #c_aki_7d = c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'].isin(['0', '1'])] 
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    
    df_save = pd.merge(info_save, c_aki_7d, how='inner', on='ICUSTAY_ID')
    df_save.columns = map(str.upper, df_save.columns)
    icustays_data = [frame for season, frame in df_save.groupby(['ICUSTAY_ID'])]
       
    count_ckd_normal = 0
    count_ckd_aki = 0
    count_akibefore_normal = 0
    count_akibefore_aki = 0
    count_normal = 0
    count_aki = 0
    count_renalfailure_normal = 0
    count_renalfailure_aki = 0
       
    for temp in icustays_data:
        
        temp = temp.sort_values(by=['ICUSTAY_ID'])
        
        gender = temp['GENDER'].values[0]
        age = temp['AGE'].values[0]
        eth = temp['ETHNICITY'].values[0]
        cr = temp['CREAT'].values[0]
         
        eGFR = caculate_eGFR_MDRD_equation(cr=cr, gender=gender, age=age, eth=eth)
                
        df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'EGFR'] = eGFR
        df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = c_aki_7d.loc[c_aki_7d['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0])]['AKI_7DAY'].values[0]
      
        if (df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
            count_aki = count_aki +1
        else:
            count_normal = count_normal + 1
              
        if (check_CKD(temp['HADM_ID']) == True):
            df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = 2
            if (info_save.loc[info_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
                count_ckd_aki = count_ckd_aki + 1
            else:
                count_ckd_normal = count_ckd_normal + 1
        
        if (check_AKI_before(temp['HADM_ID']) == True):
            df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = 3
            if (info_save.loc[info_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
                count_akibefore_aki = count_akibefore_aki + 1
            else:
                count_akibefore_normal = count_akibefore_normal + 1
        
        if (check_renal_failure(temp['HADM_ID']) == True):
            df_save.loc[df_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'] = 4
            if (info_save.loc[info_save['ICUSTAY_ID'] == int(temp['ICUSTAY_ID'].values[0]), 'AKI'].values[0] == 1):
                count_renalfailure_aki = count_renalfailure_aki + 1
            else:
                count_renalfailure_normal = count_renalfailure_normal + 1
        
    lab = pd.read_csv(os.path.join(path_preprocess_data,'labstay.csv'))
    lab.columns = map(str.upper, lab.columns)
    info_save = pd.merge(df_save, lab, how='left', on='ICUSTAY_ID')
    info_save = info_save.drop(columns=['UNNAMED: 0_x','UNNAMED: 0_y'])
    info_save = info_save.rename(columns={'SUBJECT_ID_X': 'SUBJECT_ID', 'HADM_ID_x':'HADM_ID'})
  
    chart = pd.read_csv(os.path.join(path_preprocess_data,'chart_vitals_stay.csv'))
    chart.columns = map(str.upper, chart.columns)
    df_save = pd.merge(info_save, chart, how='left', on='ICUSTAY_ID')
    df_save = df_save.drop(columns=['UNNAMED: 0', 'HADM_ID_y', 'HADM_ID_y', 'SUBJECT_ID_Y', 'SUBJECT_ID_y'])
    df_save = df_save.rename(columns={'SUBJECT_ID_X': 'SUBJECT_ID', 'HADM_ID_x':'HADM_ID'})

    comorbidities = pd.read_csv(os.path.join(path_preprocess_data,'comorbidities.csv'))
    comorbidities.columns = map(str.upper, comorbidities.columns)
    info_save = pd.merge(df_save, comorbidities, how='left', on='HADM_ID')
    info_save = info_save.drop(columns=['UNNAMED: 0'])
       
    print('NORMAL Patients in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1 within 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 1]['ICUSTAY_ID'].count()))  
    print('CKD counted as normal: {}'.format(count_ckd_normal))
    print('CKD counted as aki: {}'.format(count_ckd_aki))
    print('AKI on admission counted as normal: {}'.format(count_akibefore_normal))
    print('AKI on admission counted as aki: {}'.format(count_akibefore_aki))
    print('RENAL FAILURE counted as normal: {}'.format(count_renalfailure_normal))
    print('RENAL FAILURE counted as aki: {}'.format(count_renalfailure_aki))
    print('normal: {}'.format(count_normal))
    print('aki: {}'.format(count_aki))
    
    with open(os.path.join(path_preprocess_data,'INFO_DATASET_7days_creatinine2.csv'), 'w') as f:
        info_save.to_csv(f, encoding='utf-8', header=True)
                   
def run():

    c_aki = pd.read_csv(os.path.join(path_preprocess_data,'AKI_KIDIGO_STAGES_SQL.csv'))
    c_aki.columns = map(str.upper, c_aki.columns)
    print("c_aki_full infos")
    print("Total icustays: " ,c_aki['ICUSTAY_ID'].nunique())
    print('Non AKI Patients : {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1: {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 1]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 2: {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 2]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 3: {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 3]['ICUSTAY_ID'].count()))
    print('NAN patients: {}'.format(c_aki['AKI_STAGE'].isna().sum()))
    c_aki = c_aki.dropna(subset=['AKI_STAGE'])
        
    c_aki_7d = pd.read_csv(os.path.join(path_preprocess_data,'AKI_KIDIGO_7D_SQL.csv'))
    c_aki_7d.columns = map(str.upper, c_aki_7d.columns)
    print("c_aki_7d infos")
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    print('NON AKI Patients in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1 within 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 1]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 2 in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 2]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 3 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 3]['ICUSTAY_ID'].count()))
    print('NAN patients within 7DAY: {}'.format(c_aki_7d['AKI_STAGE_7DAY'].isna().sum()))
    c_aki_7d = c_aki_7d.dropna(subset=['AKI_STAGE_7DAY'])
    
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    
    print("USING ONLY CREATININE")     
    c_aki_7d = pd.read_csv(os.path.join(path_preprocess_data,'AKI_KIDIGO_7D_SQL_CREATININE.csv'))
    c_aki_7d.columns = map(str.upper, c_aki_7d.columns)
    print("c_aki_7d creatinin only infos")
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    print('NON AKI Patients in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1 within 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 1]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 2 in 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 2]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 3 7DAY: {}'.format(c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'] == 3]['ICUSTAY_ID'].count()))
    print('NAN patients within 7DAY: {}'.format(c_aki_7d['AKI_STAGE_7DAY'].isna().sum()))
    c_aki_7d = c_aki_7d.dropna(subset=['AKI_STAGE_7DAY'])
    #c_aki_7d = c_aki_7d.loc[c_aki_7d['AKI_STAGE_7DAY'].isin(['0', '1'])] 
    print("Total icustays: " ,c_aki_7d['ICUSTAY_ID'].nunique())
    
    c_aki = pd.read_csv(os.path.join(path_preprocess_data,'AKI_KIDIGO_STAGES_SQL_CREATININE.csv'))
    c_aki.columns = map(str.upper, c_aki.columns)
    print("c_aki_full creatinine infos")
    print("Total icustays: " ,c_aki['ICUSTAY_ID'].nunique())
    print('Non AKI Patients : {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 0]['ICUSTAY_ID'].count()))
    print('AKI patients STAGE 1: {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 1]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 2: {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 2]['ICUSTAY_ID'].count()))
    print('AKI Patients STAGE 3: {}'.format(c_aki.loc[c_aki['AKI_STAGE'] == 3]['ICUSTAY_ID'].count()))
    print('NAN patients: {}'.format(c_aki['AKI_STAGE'].isna().sum()))
    c_aki = c_aki.dropna(subset=['AKI_STAGE'])
     
    get_info_admissions()
       
    get_aki_patients_7days()

    get_aki_patients_7days_creatinine()
     
    statistical_itemid_missing()

if __name__ == '__main__':
    run()

