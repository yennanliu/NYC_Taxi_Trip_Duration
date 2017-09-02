# -*- coding: utf-8 -*-

# script predict with  test dataset 

# basic library 
import pandas as pd, numpy as np
import calendar
import pickle 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# import user defined library
from script.prepare import *
from script.train import *  
from script.model import *  
from script.parameter import * 


### ================================================ ###
#run the process 


def predict():
	model = load_model()
	df_all_ = pd.read_csv('df_all_.csv')
	X_test  = df_all_[df_all_['trip_duration'].isnull()][features].values
	y_test = xgb.predict(X_test)
    df_sub = pd.DataFrame({'id': df_all_[df_all_['trip_duration'].isnull()]['id'].values,
         				   'trip_duration': np.exp(y_test)}).set_index('id')
	
	print (df_sub)
    df_sub.to_csv('~/NYC_Taxi_Trip_Duration/output/predict_output.csv')
    print ('prediction completion !')






if __name__ == '__main__':
	
	predict()









