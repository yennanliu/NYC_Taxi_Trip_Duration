# -*- coding: utf-8 -*-

# script running process : data prepare -> model train -> ..


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


def train_model():
    
    df_all, df_train, df_test  = load_data()
    #get basic features 
    df_all_ = get_time_feature(df_all)
    df_all_ = get_time_cyclic(df_all_)
    # get other features 
    df_all_ = get_geo_feature(df_all_)
    df_all_ = pca_lon_lat(df_all_)
    # get center of trip route 
    df_all_ = gat_trip_center(df_all_)
    # get lon & lat clustering 
    df_all_ = get_clustering(df_all_)
    # get avg ride count on dropoff cluster 
    df_all_ = trip_cluser_count(df_all_)
    df_all_ = get_cluser_feature(df_all_)
    # get avg feature value 
    df_all_ = get_avg_feature(df_all_)
    # label -> 0,1 
    df_all_ = label_2_binary(df_all_)
    # count trip over 60 min
    df_all_ = trip_over_60min(df_all_)
    # get log trip duration 
    df_all_['trip_duration_log'] = df_all_['trip_duration'].apply(np.log)

    # merge with OSRM data 
    frame_fastest = load_OSRM_data()
    #print (frame_fastest.head())
    df_all_ = pd.merge(left=df_all_,
                       right=frame_fastest[['id', 'total_distance', 
                                            'total_travel_time', 'number_of_steps',
                                            'right_turn','left_turn']],
                                            on='id', how='left')
    print (df_all_.head())
    # from script.parameter import  feature list 
    
    X_train = df_all_[df_all_['trip_duration'].notnull()][features].values
    y_train = df_all_[df_all_['trip_duration'].notnull()]['trip_duration_log'].values
    X_test  = df_all_[df_all_['trip_duration'].isnull()][features].values

    # train xgb  
    xgb = XGBRegressor(n_estimators=1000, max_depth=13, min_child_weight=150, 
                   subsample=0.7, colsample_bytree=0.3)
    y_test = np.zeros(len(X_test))

    for i, (train_ind, val_ind) in enumerate(KFold(n_splits=2, shuffle=True, 
                                            random_state=1989).split(X_train)):
        print('----------------------')
        print('Training model #%d' % i)
        print('----------------------')
        
        xgb.fit(X_train[train_ind], y_train[train_ind],
                eval_set=[(X_train[val_ind], y_train[val_ind])],
                early_stopping_rounds=10, verbose=25)
    print (xgb)
    # save tuneed data for prediction 
    df_all_.to_csv('df_all_.csv')
    print ('train OK ! ')
    return xgb



### ================================================ ###


if __name__ == '__main__':
    model_xgb  = train_model()
    save_model(model_xgb)










