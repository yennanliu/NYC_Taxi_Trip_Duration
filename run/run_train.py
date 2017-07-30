# -*- coding: utf-8 -*-

# script running process : data prepare -> model train -> ..


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
print (df_train)

# only take 100 data points  here 

df_train_ = basic_feature_extract(df_train.head(100))
df_train_ = get_features(df_train_)
df_train_,df_test_ = pca_lon_lat(df_train_,df_test)
df_train_ = avg_speed(df_train_)
df_train_ = clean_data(df_train_)

print (df_train_.head())
print ('Train Data Ready  ! ')

#### fitting test dataset  ####
df_train_tree_ = clean_data(df_train_)
tree_feature = ['trip_duration', 'vendor_id', 
                'passenger_count', 'pickup_longitude', 
                'pickup_latitude','dropoff_longitude',
                'dropoff_latitude','pickup_hour','pickup_month',  
                'distance_haversine','pickup_weekday',
                'distance_manhattan']
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor()
model_tree = reg_analysis(tree_model,df_train_tree_[tree_feature])
print (model_tree)

# save model 
save_model(model_tree)










