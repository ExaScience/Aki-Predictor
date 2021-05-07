
import time
import csv
import numpy as np

import os

import tensorflow as tf

import pandas as pd
import argparse

import random

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score        
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras import layers
from keras import activations

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

from keras.layers import Activation
from keras.utils import to_categorical

scaler = preprocessing.StandardScaler()
#scaler = preprocessing.MinMaxScaler()
    
def code_ethnicity(ethinicity):
   
    if (ethinicity == 'WHITE'): return 0;
    elif (ethinicity == 'UNKNOWN/NOT SPECIFIED'): return -1;
    elif (ethinicity == 'BLACK/AFRICAN AMERICAN'): return 1; 
    elif (ethinicity == 'ASIAN'): return 2;
    elif (ethinicity == 'HISPANIC OR LATINO'): return 3;
    elif (ethinicity == 'HISPANIC/LATINO - GUATEMALAN'): return 3;
    elif (ethinicity == 'OTHER'): return -1;
    elif (ethinicity == 'HISPANIC/LATINO - PUERTO RICAN'): return 3;
    elif (ethinicity == 'PATIENT DECLINED TO ANSWER'): return -1;
    elif (ethinicity == 'ASIAN - ASIAN INDIAN'): return 2;
    elif (ethinicity == 'ASIAN - VIETNAMESE'): return 2; 
    elif (ethinicity == 'MULTI RACE ETHNICITY'): return -1;
    elif (ethinicity == 'HISPANIC/LATINO - DOMINICAN'): return 3; 
    elif (ethinicity == 'WHITE - RUSSIAN'): return 0;
    elif (ethinicity == 'BLACK/AFRICAN'): return 1;
    elif (ethinicity == 'HISPANIC/LATINO - SALVADORAN'): return 3; 
    elif (ethinicity == 'UNABLE TO OBTAIN'): return -1; 
    elif (ethinicity == 'ASIAN - CHINESE'): return 2;
    elif (ethinicity == 'BLACK/HAITIAN'): return 1;
    elif (ethinicity == 'AMERICAN INDIAN/ALASKA NATIVE'): return 4;
    elif (ethinicity == 'WHITE - EASTERN EUROPEAN'): return 0;
    elif (ethinicity == 'BLACK/CAPE VERDEAN'): return 1;
    elif (ethinicity == 'ASIAN - FILIPINO'): return 2;
    elif (ethinicity == 'CARIBBEAN ISLAND'): return 5;
    elif (ethinicity == 'SOUTH AMERICAN'): return 6;
    elif (ethinicity == 'HISPANIC/LATINO - COLOMBIAN'): return 3;
    elif (ethinicity == 'WHITE - OTHER EUROPEAN'): return 0;
    elif (ethinicity == 'WHITE - BRAZILIAN'): return 0; 
    elif (ethinicity == 'PORTUGUESE'): return 3;
    elif (ethinicity == 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)'): return 3; 
    elif (ethinicity == 'ASIAN - CAMBODIAN'): return 2; 
    elif (ethinicity == 'ASIAN - THAI'): return 2;
    elif (ethinicity == 'HISPANIC/LATINO - HONDURAN'): return 3; 
    elif (ethinicity == 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'): return 5; 
    elif (ethinicity == 'HISPANIC/LATINO - CUBAN'): return 3;
    elif (ethinicity == 'MIDDLE EASTERN'): return 7; 
    elif (ethinicity == 'ASIAN - OTHER'): return 2; 
    elif (ethinicity == 'HISPANIC/LATINO - MEXICAN'): return 3;
    elif (ethinicity == 'ASIAN - KOREAN'): return 2;
    elif (ethinicity == 'ASIAN - JAPANESE'): return 2;
    elif (ethinicity == 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE'): return 4;

def code_gender(gender):
    
    if (gender =='F'): return 0;
    else:  return 1;

outdir = "/AKI_predictor/output/"
 
def cleanup_data(filename, outdir):       

    # read the data from the CSV
    df = pd.read_csv(open(filename, "r"), delimiter=",")
    df.columns = map(str.upper, df.columns)
    print(df.shape)
    
    print(df.groupby('AKI')['ICUSTAY_ID'].nunique())
    print(df.groupby('AKI_STAGE_7DAY')['ICUSTAY_ID'].nunique())
       
    #exclude CKD and AKI on admission patients
    df = df[~(df['AKI'] == 2)]
    df = df[~(df['AKI'] == 3)]
    df = df[~(df['AKI'] == 4)]
    
    print(df.groupby('AKI')['ICUSTAY_ID'].nunique())
      
    #Consider only adults
    df = df[~(df['AGE'] < 18)] 
    df['ETHNICITY'] = df['ETHNICITY'].apply(lambda x: code_ethnicity(x)) 
    df['GENDER'] = df['GENDER'].apply(lambda x: code_gender(x)) 
    
    print(df.groupby('ETHNICITY')['ICUSTAY_ID'].nunique())
    
    df = df.rename(columns={'HADM_ID_X' : 'HADM_ID','GLUCOSE_MIN_X':'GLUCOSE_MIN', 'GLUCOSE_MAX_X':'GLUCOSE_MAX', 'SUBJECT_ID_Y': 'SUBJECT_ID', 'SUBJECT_ID_X.1': 'SUBJECT_ID'})
    df = df.fillna(0)
    
   # df = df.fillna(df.mean())

    #print(pd.isna(df) == True)
                    
    df = df.drop(df.columns[1], axis=1)
    
    print("cleanup1", df.columns)
        
    if 'AKI_7DAY' in df.columns: 
        # ,  'SUBJECT_ID_x','GLUCOSE_MIN_y', 'GLUCOSE_MAX_y'
        df = df.drop(['ADMITTIME', 'DISCHTIME', 'OUTTIME', 'INTIME', 'DOB', 'CHARTTIME_CREAT', 'UNNAMED: 0', 'AKI_STAGE_CREAT', 'AKI_7DAY', 'GLUCOSE_MAX_Y', 'GLUCOSE_MIN_Y'], axis =1)
    else:
        df = df.drop(['ADMITTIME', 'DISCHTIME', 'OUTTIME', 'INTIME', 'DOB', 'CHARTTIME_CREAT', 'CHARTTIME_UO', 'HADM_ID_x', 'Unnamed: 0', 'AKI_STAGE_CREAT', 'AKI_STAGE_48HR', 'AKI_STAGE_UO', 'AKI_48HR', 'SUBJECT_ID_y', 'SUBJECT_ID_x.1', 'SUBJECT_ID_x', 'HADM_ID_y' ,'GLUCOSE_MIN_y', 'GLUCOSE_MAX_y'], axis =1)
 
    return df

def accuracy_confusion(confusion_matrix):
   
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements
   
def compute_metrics(ytest, ypred, multiclass):

    if multiclass == True:
        lb = preprocessing.LabelBinarizer()
        lb.fit(ytest)
        ytest = lb.transform(ytest)
        ypred = lb.transform(ypred)
        print(classification_report(ytest, ypred)) #, labels=[0, 1, 2, 3]))
        C = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1))
    else:
        print(classification_report(ytest, ypred))
         
        C = confusion_matrix(ytest, ypred)
        
    normed_C = normalize(C, axis=1, norm='l1')
         
    print ("confusion_matrix", C)  
    print("accuracy confusion matrix" , accuracy_confusion(C))  
    print("Area Under ROC:", metrics.roc_auc_score(ytest, ypred))
    print("Accuracy score ", accuracy_score(ytest, ypred))

def aki_model(X, Y, X_test, Y_test, filename):
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, input_dim = X.shape[1] , activation = 'relu'),    
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(Y.shape[1], activation=tf.nn.softmax),
        ])
          
    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X, Y, epochs=1000, callbacks=[callback], verbose=0) 
    
    model.save_weights(filename + ".weights")
    
    y_pred_test = model.predict_classes(X_test)
    y_pred_train = model.predict_classes(X)
  
    return y_pred_train, y_pred_test

def make_train_test(df, ntest, seed = None):
  
    if ntest < 1:
        ntest = df.shape[0] * ntest
    if seed is not None:
        np.random.seed(seed)
    
    ntest = int(round(ntest))
    rperm = np.random.permutation(df.shape[0])
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    dftrain = df.iloc[train]
    dftest  = df.iloc[test]
    
    return dftrain, dftest
   
def run_aki_model(df, filename):
    
    scaler.fit(df.iloc[:,2:]) 
    df.iloc[:,2:] = scaler.transform(df.iloc[:,2:])
     
    X = df.drop(['AKI_STAGE_7DAY', 'AKI'], axis=1)
    Y = df['AKI_STAGE_7DAY'] 
    
    df_train, df_test = make_train_test(df, 0.2, seed=1234)
    
    X_train = df_train.drop(['AKI_STAGE_7DAY', 'AKI'], axis=1)
    Y_train = df_train['AKI_STAGE_7DAY']
    
    X_test = df_test.drop(['AKI_STAGE_7DAY', 'AKI'], axis=1)
    Y_test = df_test['AKI_STAGE_7DAY']
                    
    Y_pred_train, Y_pred_test = aki_model(X_train, to_categorical(Y_train), X_test, to_categorical(Y_test), filename)
    
    compute_metrics(Y_test, Y_pred_test, multiclass=True)
        
def cluster_ethnicity(df):
    
    print("start ethinicty clustering")
    
    caucasian = df.loc[df['ETHNICITY']==0]
    african = df.loc[df['ETHNICITY']==1]
    hispanic = df.loc[df['ETHNICITY']==3]
    others_non_caucasian =  df.loc[df['ETHNICITY']!=0]
       
    scaler.fit(caucasian.iloc[:,2:]) 
    caucasian.iloc[:,2:] = scaler.transform(caucasian.iloc[:,2:])
    african.iloc[:,2:] = scaler.transform(african.iloc[:,2:])
    hispanic.iloc[:,2:] = scaler.transform(hispanic.iloc[:,2:])
    others_non_caucasian.iloc[:,2:] = scaler.transform(others_non_caucasian.iloc[:,2:])
    
    X_caucasian = caucasian.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_caucasian = caucasian['AKI_STAGE_7DAY']

    X_african = african.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_african = african['AKI_STAGE_7DAY']

    X_hispanic = hispanic.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_hispanic = hispanic['AKI_STAGE_7DAY']
        
    X_others_non_caucasian = others_non_caucasian.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_others_non_caucasian = others_non_caucasian['AKI_STAGE_7DAY']
    
    print("Y_caucasian" , np.unique(Y_caucasian, return_counts=True))
    print("Y_african" , np.unique(Y_african, return_counts=True))
    print("Y_hispanic" , np.unique(Y_hispanic, return_counts=True))
    print("Y_others_non_caucasian" , np.unique(Y_others_non_caucasian, return_counts=True))
  
    print("Train on TRAIN caucasian set + test on TEST caucasian set + test on all african set")
       
    caucasian_train, caucasian_test = make_train_test(caucasian, 0.2, seed=1234)
    
    X_caucasian_train = caucasian_train.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_caucasian_train = caucasian_train['AKI_STAGE_7DAY']
   
    X_caucasian_test = caucasian_test.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_caucasian_test = caucasian_test['AKI_STAGE_7DAY']
 
    african_train, african_test = make_train_test(african, 0.2, seed=1234)
    
    X_african_train = african_train.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_african_train = african_train['AKI_STAGE_7DAY']
   
    X_african_test = african_test.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_african_test = african_test['AKI_STAGE_7DAY']
 
    hispanic_train, hispanic_test = make_train_test(hispanic, 0.2, seed=1234)
    
    X_hispanic_train = hispanic_train.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_hispanic_train = hispanic_train['AKI_STAGE_7DAY']
   
    X_hispanic_test = hispanic_test.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_hispanic_test = hispanic_test['AKI_STAGE_7DAY']
  
    others_non_caucasian_train, others_non_caucasian_test = make_train_test(others_non_caucasian, 0.2, seed=1234)
    
    X_others_non_caucasian_train = others_non_caucasian_train.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_others_non_caucasian_train = others_non_caucasian_train['AKI_STAGE_7DAY']
   
    X_others_non_caucasian_test = others_non_caucasian_test.drop(['AKI', 'AKI_STAGE_7DAY'], axis=1)
    Y_others_non_caucasian_test = others_non_caucasian_test['AKI_STAGE_7DAY']

  
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, input_dim = X_caucasian_train.shape[1] , activation = 'relu'),    
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(to_categorical(Y_caucasian_train).shape[1], activation=tf.nn.softmax),
        ])
    #     
    #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_caucasian_train, to_categorical(Y_caucasian_train), epochs=1000, callbacks=[callback], verbose=0) # Run with 1 epoch to speed things up for demo purposes
    
    Y_caucasian_test_pred = model.predict_classes(X_caucasian_test)
    Y_others_non_caucasian_pred = model.predict_classes(X_others_non_caucasian)
    Y_african_test_pred = model.predict_classes(X_african_test)
    Y_hispanic_test_pred = model.predict_classes(X_hispanic_test)
    
    compute_metrics(Y_caucasian_test, Y_caucasian_test_pred, multiclass=True)
    compute_metrics(Y_others_non_caucasian, Y_others_non_caucasian_pred, multiclass=True)  
    compute_metrics(Y_african_test, Y_african_test_pred, multiclass=True)
    compute_metrics(Y_hispanic_test, Y_hispanic_test_pred, multiclass=True)  

def split_randomly(df, ratio, str):
        
    scaler.fit(df.iloc[:,2:]) 
    
    df.iloc[:,2:] = scaler.transform(df.iloc[:,2:])
   
    df_train, df_test = make_train_test(df, ratio, seed=1234)
    
    X_train = df_train.drop(['AKI_STAGE_7DAY', 'AKI'], axis=1)
    Y_train = df_train['AKI_STAGE_7DAY']
 
    X_test = df_test.drop(['AKI_STAGE_7DAY', 'AKI'], axis=1)
    Y_test = df_test['AKI_STAGE_7DAY']
    
    Y_pred_train, Y_pred_test = aki_model(X_train, to_categorical(Y_train), X_test, to_categorical(Y_test), str)
    
    compute_metrics(Y_test, Y_pred_test, multiclass=True)
       
def change_data_size(df):
    
    print("random subsampling")
       
    split_randomly(df, 0.02, "2% test size")
    split_randomly(df, 0.05, "5% test size")
    split_randomly(df, 0.10, "10% test size")
    split_randomly(df, 0.20, "20% test size")
    split_randomly(df, 0.40, "40% test size")
    split_randomly(df, 0.50, "50% test size")
    split_randomly(df, 0.60, "60% test size")
    split_randomly(df, 0.80, "80% test size")
    split_randomly(df, 0.90, "90% test size")
    split_randomly(df, 0.95, "95% test size")
               
if __name__ == '__main__':
            
    df = cleanup_data("INFO_DATASET_7days_creatinine_withcomorbidities.csv", outdir)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
 
    df = df[['AKI', 'AKI_STAGE_7DAY', 'CREATININE_MAX', 'CREATININE_MIN', 'CREAT', 'EGFR', 'POTASSIUM_MAX', 'GLUCOSE_MAX', 'PLATELET_MIN', 
               'BUN_MAX', 'WBC_MIN', 'PLATELET_MAX', 'TEMPC_MEAN', 'GLUCOSE_MEAN', 'PTT_MAX', 'TEMPC_MIN', 'BUN_MIN', 'HEMATOCRIT_MIN', 
               'SPO2_MEAN', 'MEANBP_MEAN', 'AGE', 'HEARTRATE_MEAN', 'PT_MAX', 'TEMPC_MAX', 'RESPRATE_MEAN', 'CHLORIDE_MAX', 'GLUCOSE_MIN', 'WBC_MAX', 
               'DIASBP_MEAN', 'SYSBP_MAX', 'DIASBP_MIN', 'CHLORIDE_MIN', 'SPO2_MIN', 'HEARTRATE_MAX', 'HEMOGLOBIN_MAX', 'SYSBP_MEAN', 'HEMATOCRIT_MAX', 'DIASBP_MAX', 
               'HEARTRATE_MIN', 'SYSBP_MIN', 'SODIUM_MIN', 'MEANBP_MAX', 'BICARBONATE_MAX', 'MEANBP_MIN', 'SODIUM_MAX', 'ANIONGAP_MAX', 'ANIONGAP_MIN', 'HEMOGLOBIN_MIN', 
               'LACTATE_MIN', 'BICARBONATE_MIN', 'PTT_MIN', 'PT_MIN', 'BILIRUBIN_MAX', 'RESPRATE_MIN', 'LACTATE_MAX', 'RESPRATE_MAX', 'ALBUMIN_MIN', 'POTASSIUM_MIN', 'INR_MAX', 
               'ALBUMIN_MAX', 'BILIRUBIN_MIN', 'INR_MIN', 'BANDS_MIN', 'ETHNICITY', 'BANDS_MAX', 'HYPERTENSION', 'DIABETES_UNCOMPLICATED', 'VALVULAR_DISEASE', 
               'CONGESTIVE_HEART_FAILURE', 'SPO2_MAX', 'ALCOHOL_ABUSE', 'GENDER', 'CARDIAC_ARRHYTHMIAS', 'PERIPHERAL_VASCULAR', 'OBESITY', 'HYPOTHYROIDISM', 'DIABETES_COMPLICATED', 
               'LIVER_DISEASE', 'DRUG_ABUSE', 'RENAL_FAILURE']]
    
    df2 = cleanup_data("INFO_DATASET_7days_creatinine+urine_withcomorbidities.csv", outdir)
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
 
    df2 = df2[['AKI', 'AKI_STAGE_7DAY', 'UO_RT_24HR', 'UO_RT_12HR', 'UO_RT_6HR', 'CREATININE_MAX', 'CREATININE_MIN', 'CREAT', 'EGFR', 'PLATELET_MAX', 'WBC_MAX', 'BUN_MAX', 
        'PLATELET_MIN', 'AGE', 'GLUCOSE_MIN', 'TEMPC_MIN', 'WBC_MIN', 'GLUCOSE_MAX', 'DIASBP_MEAN', 'BUN_MIN', 'RESPRATE_MEAN', 'SYSBP_MAX', 'POTASSIUM_MAX', 'CHLORIDE_MAX', 
        'HEARTRATE_MAX', 'HEARTRATE_MEAN', 'SPO2_MEAN', 'PTT_MAX', 'MEANBP_MEAN', 'CHLORIDE_MIN', 'GLUCOSE_MEAN', 'PTT_MIN', 'TEMPC_MAX', 'MEANBP_MIN', 'ANIONGAP_MAX', 'SODIUM_MIN',
       'HEMOGLOBIN_MAX', 'HEMATOCRIT_MAX', 'MEANBP_MAX', 'DIASBP_MAX', 'HEMATOCRIT_MIN', 'SPO2_MIN', 'SODIUM_MAX', 'TEMPC_MEAN', 'DIASBP_MIN',
       'HEARTRATE_MIN', 'RESPRATE_MAX', 'HEMOGLOBIN_MIN', 'BICARBONATE_MAX', 'SYSBP_MIN', 'SYSBP_MEAN', 'BICARBONATE_MIN', 'POTASSIUM_MIN',
       'BILIRUBIN_MAX', 'LACTATE_MAX', 'ANIONGAP_MIN', 'ALBUMIN_MAX', 'PT_MIN', 'BILIRUBIN_MIN', 'INR_MIN', 'ALBUMIN_MIN', 'RESPRATE_MIN', 'PT_MAX',
       'ETHNICITY', 'LACTATE_MIN', 'INR_MAX', 'BANDS_MAX', 'BANDS_MIN', 'HYPERTENSION', 'HYPOTHYROIDISM', 'CONGESTIVE_HEART_FAILURE', 'GENDER', 'CARDIAC_ARRHYTHMIAS', 'SPO2_MAX',
       'ALCOHOL_ABUSE', 'DRUG_ABUSE', 'VALVULAR_DISEASE', 'OBESITY', 'PERIPHERAL_VASCULAR', 'DIABETES_COMPLICATED', 'LIVER_DISEASE', 'DIABETES_UNCOMPLICATED', 'RENAL_FAILURE']]
    
          
    run_aki_model(df, "creatinine_model")
    run_aki_model(df2, "creatinine_urine_model")
      
    cluster_ethnicity(df)
    cluster_ethnicity(df2)
    
    change_data_size(df)
    change_data_size(df2)
                  