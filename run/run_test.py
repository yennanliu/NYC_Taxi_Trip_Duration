# -*- coding: utf-8 -*-

# script predict with  test dataset 


# basic library 
import pandas as pd, numpy as np
import calendar
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import pickle 


# import user defined library
from script.prepare import *
from script.train import * 
from script.model import *  


### ================================================ ###
#run the process 

df_train, df_test, sampleSubmission = load_data()
df_test_ = basic_feature_extract(df_test)
df_test_ = get_features(df_test_)
#df_train_,df_test_ = pca_lon_lat(df_train_,df_test_)
#df_train_ = avg_speed(df_train_)
df_test_ = clean_data(df_test_)
print (df_test.head())


# prepare submit data 
submit = pd.DataFrame()
submit['id'] = df_test_['id']
submit_feature_ = ['vendor_id', 
                'passenger_count', 'pickup_longitude', 
                'pickup_latitude','dropoff_longitude',
                'dropoff_latitude','pickup_hour','pickup_month',  
                'distance_haversine','pickup_weekday',
                'distance_manhattan']
# load model 
model = load_model()
print (model)
X_testdata = df_test_[submit_feature_]
# predict
submit['trip_duration'] = model.predict(X_testdata)
print ('predict finish !')
# set id as index
submit= submit.set_index('id')

print (submit.head())
#print (shape(submit))
# save output 
submit.to_csv('~/NYC_Taxi_Trip_Duration/output/submit0726.csv')
print ('predict data saved !')












