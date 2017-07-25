# -*- coding: utf-8 -*-

# script running process : data prepare -> model train -> ..


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
print (df_train)

# only take 100 data points  here 
df_train_ = basic_feature_extract(df_train.head(100))
df_train_.loc[:, 'distance_haversine'] = get_haversine_distance(
                      df_train_['pickup_latitude'].values,
                      df_train_['pickup_longitude'].values,
                      df_train_['dropoff_latitude'].values,
                      df_train_['dropoff_longitude'].values)
df_train_.loc[:, 'distance_manhattan'] = get_manhattan_distance(
                      df_train_['pickup_latitude'].values,
                      df_train_['pickup_longitude'].values,
                      df_train_['dropoff_latitude'].values,
                      df_train_['dropoff_longitude'].values)

df_train_ = clean_data(df_train_)

print (df_train_.head())
print ('Train Data Ready  ! ')


