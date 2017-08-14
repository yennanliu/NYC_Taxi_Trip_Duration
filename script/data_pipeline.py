# -*- coding: utf-8 -*-


### ================================================ ###

"""
# basic library 
import pandas as pd, numpy as np
import calendar
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import pickle 
# import user defined library
from prepare import *
from train import *  
from model import *  
df_train, df_test, sampleSubmission = load_data()
#print (df_train.head())
# only take 100 data points  here 
df_train_ = get_time_feature(df_train.head(100))
df_train_ = get_features(df_train_)
df_train_,df_test_ = pca_lon_lat(df_train_,df_test)
#df_train_ = avg_speed(df_train_)
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
X = df_train_tree_[tree_feature]
y = df_train_tree_['trip_duration']

"""


### ================================================ ###

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline


# your dataset here !  (X,y)

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) rf
rf  = RandomForestRegressor()
# 3) fitting 
anova_svm = make_pipeline(anova_filter, rf)
anova_svm.fit(X, y)
anova_svm.predict(X)
print (anova_svm.predict(X))



















