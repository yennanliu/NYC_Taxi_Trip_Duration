# -*- coding: utf-8 -*-

# script for data preparation 


# basic library 
import pandas as pd, numpy as np
import calendar


# basic feature extract 
def basic_feature_extract(df):
    df_= df.copy()
    # pickup
    df_["pickup_date"] = pd.to_datetime(df_.pickup_datetime.apply(lambda x : x.split(" ")[0]))
    df_["pickup_hour"] = df_.pickup_datetime.apply(lambda x : x.split(" ")[1].split(":")[0])
    df_["pickup_year"] = df_.pickup_datetime.apply(lambda x : x.split(" ")[0].split("-")[0])
    df_["pickup_month"] = df_.pickup_datetime.apply(lambda x : x.split(" ")[0].split("-")[1])
    df_["pickup_weekday"] = df_.pickup_datetime.apply(lambda x :pd.to_datetime(x.split(" ")[0]).weekday())
    # dropoff 
    df_["dropoff_date"] = pd.to_datetime(df_.dropoff_datetime.apply(lambda x : x.split(" ")[0]))
    df_["dropoff_hour"] = df_.dropoff_datetime.apply(lambda x : x.split(" ")[1].split(":")[0])
    df_["dropoff_year"] = df_.dropoff_datetime.apply(lambda x : x.split(" ")[0].split("-")[0])
    df_["dropoff_month"] = df_.dropoff_datetime.apply(lambda x : x.split(" ")[0].split("-")[1])
    df_["dropoff_weekday"] = df_.dropoff_datetime.apply(lambda x :pd.to_datetime(x.split(" ")[0]).weekday())
    # get weekday
    list(calendar.day_name)
    df_['pickup_week_'] = pd.to_datetime(df_train.pickup_datetime,coerce=True).dt.weekday
    df_['pickup_weekday_'] = df_['pickup_week_'].apply(lambda x: calendar.day_name[x])
    return df_

# feature engineering 
def get_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h 

def get_manhattan_distance(lat1, lng1, lat2, lng2):
    a = get_haversine_distance(lat1, lng1, lat1, lng2)
    b = get_haversine_distance(lat1, lng1, lat2, lng1)
    return a + b

# data cleaning
def clean_data(df):
    df_ = df.copy()
    # remove possible outlier in haversine distance 
    df_ = df_[(df_['distance_haversine'] < df_['distance_haversine'].quantile(0.95))&
         (df_['distance_haversine'] > df_['distance_haversine'].quantile(0.05))]
    # remove possible outlier in trip duration 
    df_ = df_[(df_['trip_duration'] < df_['trip_duration'].quantile(0.95))&
         (df_['trip_duration'] > df_['trip_duration'].quantile(0.05))]
    return df_

# load data
def load_data():
	df_train = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/train.csv')
	df_test = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/test.csv')
	sampleSubmission = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/sample_submission.csv')
	return df_train, df_test, sampleSubmission

### ================================================ ###


df_train, df_test, sampleSubmission = load_data()
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

