# -*- coding: utf-8 -*-

# script fitting test dataset 


# basic library 
import pandas as pd, numpy as np
import calendar
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
# import user defined library
from prepare import *
from train import * 


### ================================================ ###
#run the process 

df_train, df_test, sampleSubmission = load_data()
print (df_test.head())
df_test_ = basic_feature_extract_(df_test)
df_test_ = get_weekday(df_test_)
# get haversine distance 
df_test_.loc[:, 'distance_haversine'] = get_haversine_distance(
                      df_test_['pickup_latitude'].values,
                      df_test_['pickup_longitude'].values,
                      df_test_['dropoff_latitude'].values,
                      df_test_['dropoff_longitude'].values)

# get manhattan distance 
df_test_.loc[:, 'distance_manhattan'] = get_manhattan_distance(
                      df_test_['pickup_latitude'].values,
                      df_test_['pickup_longitude'].values,
                      df_test_['dropoff_latitude'].values,
                      df_test_['dropoff_longitude'].values)
# prepare submit data 
submit = pd.DataFrame()
submit['id'] = df_test_['id']
submit_feature_ = ['vendor_id', 
                'passenger_count', 'pickup_longitude', 
                'pickup_latitude','dropoff_longitude',
                'dropoff_latitude','pickup_hour','pickup_month',  
                'distance_haversine','pickup_weekday',
                'distance_manhattan']

X_testdata = df_test_[tree_feature_]
# predict
submit['trip_duration'] = model_tree.predict(X_testdata)
# set id as index
submit= submit.set_index('id')

print (submit.head())
print (shape(submit))
print ('predict finish !')
# save output 
submit.to_csv('~/NYC_Taxi_Trip_Duration/output/submit0726.csv')











