# -*- coding: utf-8 -*-
# load basics library 
import pandas as pd, numpy as np
import calendar
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

# feature engineering 
def get_time_feature(df):
    df_= df.copy()
    # pickup
    df_['pickup_datetime'] = pd.to_datetime(df_.pickup_datetime)
    df_.loc[:, 'pickup_date'] = df_['pickup_datetime'].dt.date
    df_.loc[:, 'pickup_weekday'] = df_['pickup_datetime'].dt.weekday
    df_.loc[:, 'pickup_day'] = df_['pickup_datetime'].dt.day
    df_.loc[:, 'pickup_month'] = df_['pickup_datetime'].dt.month
    df_.loc[:, 'pickup_year'] = df_['pickup_datetime'].dt.year 
    df_.loc[:, 'pickup_hour'] = df_['pickup_datetime'].dt.hour
    df_.loc[:, 'pickup_minute'] = df_['pickup_datetime'].dt.minute
    df_.loc[:,'weekofyear'] = df_['pickup_datetime'].dt.weekofyear
    df_.loc[:,'week_delta'] = df_['pickup_datetime'].dt.weekday + \
                        ((df_['pickup_datetime'].dt.hour + \
                        (df_['pickup_datetime'].dt.minute / 60.0)) / 24.0)
    df_.loc[:, 'pickup_time_delta'] = (df_['pickup_datetime'] - df_['pickup_datetime'].min()).map(
                                     lambda x: x.total_seconds())
    # dropoff
    # in case test data have no dropoff_datetime 
    try:
        df_['dropoff_datetime'] = pd.to_datetime(df_.dropoff_datetime)
        df_.loc[:, 'dropoff_date'] = df_['dropoff_datetime'].dt.date
        df_.loc[:, 'dropoff_weekday'] = df_['dropoff_datetime'].dt.weekday
        df_.loc[:, 'dropoff_day'] = df_['dropoff_datetime'].dt.day
        df_.loc[:, 'dropoff_month'] = df_['dropoff_datetime'].dt.month
        df_.loc[:, 'dropoff_year'] = df_['dropoff_datetime'].dt.year 
        df_.loc[:, 'dropoff_hour'] = df_['dropoff_datetime'].dt.hour
        df_.loc[:, 'dropoff_minute'] = df_['dropoff_datetime'].dt.minute
        df_.loc[:, 'dropoff_time_delta'] = (df_['dropoff_datetime'] - df_['dropoff_datetime'].min()).map(
                                            lambda x: x.total_seconds())
    except:
        pass 
    return df_

# make weekday and hour cyclic, since we want to let machine understand 
# these features are in fact periodically 
def get_time_cyclic(df):
    df_ = df.copy()
    df_.pickup_hour = df_.pickup_hour.astype('int')
    df_['week_delta_sin'] = np.sin((df_['week_delta'] / 7) * np.pi)**2
    df_['week_delta_cos'] = np.cos((df_['week_delta'] / 7) * np.pi)**2
    df_['pickup_hour_sin'] = np.sin((df_['pickup_hour'] / 24) * np.pi)**2
    df_['pickup_hour_cos'] = np.cos((df_['pickup_hour'] / 24) * np.pi)**2
    return df_

# Haversine distance
def get_haversine_distance(lat1, lng1, lat2, lng2):
    # km
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  #  km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h 

# Manhattan distance
# Taxi cant fly ! have to move in blocks/roads
def get_manhattan_distance(lat1, lng1, lat2, lng2):
    # km 
    a = get_haversine_distance(lat1, lng1, lat1, lng2)
    b = get_haversine_distance(lat1, lng1, lat2, lng1)
    return a + b

# get direction (arc tangent angle)
def get_direction(lat1, lng1, lat2, lng2):
    # theta
    AVG_EARTH_RADIUS = 6371  #  km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def gat_trip_center(df):
    df_ = df.copy()
    df_.loc[:, 'center_latitude'] = (df_['pickup_latitude'].values + df_['dropoff_latitude'].values) / 2
    df_.loc[:, 'center_longitude'] = (df_['pickup_longitude'].values + df_['dropoff_longitude'].values) / 2
    return df_

# PCA to transform longitude and latitude
# to improve decision tree performance 
from sklearn.decomposition import PCA
def pca_lon_lat(df):
    df_ =df.copy()
    X = np.vstack \
            ((df_[['pickup_latitude', 'pickup_longitude']].values,
              df_[['dropoff_latitude', 'dropoff_longitude']].values))
    # remove potential lon & lat outliers 
    min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
    max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
    X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]
    pca = PCA().fit(X)
    df_['pickup_pca0'] = pca.transform(df_[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df_['pickup_pca1'] = pca.transform(df_[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df_['dropoff_pca0'] = pca.transform(df_[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df_['dropoff_pca1'] = pca.transform(df_[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    # manhattan distance from pca lon & lat 
    df_.loc[:, 'pca_manhattan'] = np.abs(df_['dropoff_pca1'] - df_['pickup_pca1']) + np.abs(df_['dropoff_pca0'] - df_['pickup_pca0'])
    return df_ 

# get lon & lat clustering for following avg location speed calculation
def get_clustering(df):
    df_ = df.copy()
    coords = np.vstack((df_[['pickup_latitude', 'pickup_longitude']].values,
                        df_[['dropoff_latitude', 'dropoff_longitude']].values))
    df_ = df.copy()
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=40, batch_size=10000).fit(coords[sample_ind])
    df_.loc[:, 'pickup_cluster'] = kmeans.predict(df_[['pickup_latitude', 'pickup_longitude']])
    df_.loc[:, 'dropoff_cluster'] = kmeans.predict(df_[['dropoff_latitude', 'dropoff_longitude']])
    return df_


def trip_cluser_count(df):
    df_ = df.copy()
    df_.pickup_datetime = pd.to_datetime(df_.pickup_datetime)
    group_freq = '60min'
    df_dropoff_counts = df_ \
        .set_index('pickup_datetime') \
        .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('dropoff_cluster').rolling('240min').mean() \
        .drop('dropoff_cluster', axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})
        
    df_['pickup_datetime_group'] = df_['pickup_datetime'].dt.round(group_freq)
    df_['dropoff_cluster_count'] = \
            df_[['pickup_datetime_group', 'dropoff_cluster']]\
            .merge(df_dropoff_counts,on=['pickup_datetime_group', 'dropoff_cluster'], how='left')\
            ['dropoff_cluster_count'].fillna(0)
            
    return df_

def avg_cluster_speed_(df):
    df_ = df.copy()
    # only get pickup_cluster first as test here 
    for gby_col in ['pickup_cluster']:
        gby = df_.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'trip_duration']]
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        df_ = pd.merge(df_, gby, how='left', left_on=gby_col, right_index=True)
        #df_test = pd.merge(df_test, gby, how='left', left_on=gby_col, right_index=True)
    for gby_cols in [
                 ['pickup_cluster', 'dropoff_cluster']]:
        coord_speed = df_.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
        coord_count = df_.groupby(gby_cols).count()[['id']].reset_index()
        coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
        #coord_stats = coord_stats[coord_stats['id'] > 100]
        coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
        df_ = pd.merge(df_, coord_stats, how='left', on=gby_cols)
    return df_

def get_cluser_feature(df):
    df_ = df.copy()
    # avg cluster speed  
    avg_cluser_h = df_.groupby(['pickup_cluster','dropoff_cluster']).mean()['avg_speed_h'].reset_index()
    avg_cluser_h.columns = ['pickup_cluster','dropoff_cluster','avg_speed_cluster_h']
    avg_cluser_m = df_.groupby(['pickup_cluster','dropoff_cluster']).mean()['avg_speed_m'].reset_index()
    avg_cluser_m.columns = ['pickup_cluster','dropoff_cluster','avg_speed_cluster_m']
    # merge 
    df_ = pd.merge(df_,avg_cluser_h, how = 'left', on = ['pickup_cluster','dropoff_cluster'])
    df_ = pd.merge(df_,avg_cluser_m, how = 'left', on = ['pickup_cluster','dropoff_cluster'])
    # avg cluster duration 
    avg_cluser_duration = df_.groupby(['pickup_cluster','dropoff_cluster']).mean()['trip_duration'].reset_index()
    avg_cluser_duration.columns = ['pickup_cluster','dropoff_cluster','avg_cluster_duration']
    # merge 
    df_ = pd.merge(df_,avg_cluser_duration, how = 'left', on = ['pickup_cluster','dropoff_cluster'])
    
    return df_

def get_avg_feature(df):
    df_ = df.copy()
    ################################################################
    # using following tricks we can get average duration, speed as #
    # features which are not available at first                    #
    # since duration is the values we want to predict              #
    ################################################################
    
    # avg  duration
    avg_duration = df_.groupby(['pickup_weekday','pickup_hour']).mean()['trip_duration'].reset_index()
    avg_duration.columns = ['pickup_weekday','pickup_hour','avg_trip_duration']
    # avg speed 
    avg_speed = df_.groupby(['pickup_weekday','pickup_hour']).mean()['avg_speed_h'].reset_index()
    avg_speed.columns = ['pickup_weekday','pickup_hour','avg_trip_speed_h']
    # avg month
    month_avg = df_.groupby('pickup_month').trip_duration.mean().reset_index()
    month_avg.columns = ['pickup_month', 'month_avg']
    # avg weekofyear
    week_year_avg = df_.groupby('weekofyear').trip_duration.mean().reset_index()
    week_year_avg.columns = ['weekofyear', 'week_of_year_avg']
    # avg day of month
    day_month_avg = df_.groupby('pickup_date').trip_duration.mean().reset_index()
    day_month_avg.columns = ['pickup_date', 'day_of_month_avg']
    # avg hour
    hour_avg = df_.groupby('pickup_hour').trip_duration.mean().reset_index()
    hour_avg.columns = ['pickup_hour', 'hour_avg']
    # avg pickup weekday
    day_week_avg = df_.groupby('pickup_weekday').trip_duration.mean().reset_index()
    day_week_avg.columns = ['pickup_weekday', 'day_week_avg']
    # merge 
    df_ = pd.merge(df_,avg_duration, how = 'left', on = ['pickup_weekday','pickup_hour'])
    df_ = pd.merge(df_,avg_speed, how = 'left', on = ['pickup_weekday','pickup_hour'])
    df_ = pd.merge(df_,month_avg,how='left',on='pickup_month')
    df_ = pd.merge(df_, week_year_avg,how='left',on='weekofyear')
    df_ = pd.merge(df_, day_month_avg,how='left' ,on='pickup_date')
    df_ = pd.merge(df_, hour_avg, how='left' ,on='pickup_hour')
    df_ = pd.merge(df_, day_week_avg,how='left',on='pickup_weekday') 
    
    return df_

def label_2_binary(df):
    df_ = df.copy()
    df_['store_and_fwd_flag_'] = df_['store_and_fwd_flag'].map(lambda x: 0 if x =='N' else 1)
    return df_

def trip_over_60min(df):
    df_ = df.copy()
    df_counts = df_.set_index('pickup_datetime')[['id']].sort_index()
    df_counts['count_60min'] = df_counts.isnull().rolling('60min').count()['id']
    df_ = df_.merge(df_counts, on='id', how='left')
    return df_ 

def get_geo_feature(df):
    # km 
    df_ = df.copy()
    ###  USING .loc making return array ordering 
    # distance
    df_.loc[:, 'distance_haversine'] = get_haversine_distance(
                                      df_['pickup_latitude'].values,
                                      df_['pickup_longitude'].values,
                                      df_['dropoff_latitude'].values,
                                      df_['dropoff_longitude'].values)
    df_.loc[:, 'distance_manhattan'] = get_manhattan_distance(
                                      df_['pickup_latitude'].values,
                                      df_['pickup_longitude'].values,
                                      df_['dropoff_latitude'].values,
                                      df_['dropoff_longitude'].values)
    # direction 
    df_.loc[:, 'direction'] = get_direction(df_['pickup_latitude'].values,
                                          df_['pickup_longitude'].values, 
                                          df_['dropoff_latitude'].values, 
                                          df_['dropoff_longitude'].values)
    # Get Average driving speed 
    # km/hr
    # (km/sec = 3600 * (km/hr))
    # in case trip duration is not available in test dataset 
    try:
        df_.loc[:, 'avg_speed_h'] = 3600 * df_['distance_haversine'] / df_['trip_duration']
        df_.loc[:, 'avg_speed_m'] = 3600 * df_['distance_manhattan'] / df_['trip_duration']
    except:
        pass
    
    return df_

def get_label_feature(df):
    df_ = df.copy()
    # weekday or weekend 
    df_['weekend'] = df_.pickup_weekday.map(lambda x : 1 if x== 5 or x==6 else 0 )
    # pickup hour 6-9  
    df_['hr_6_9'] = df_.pickup_hour.map(lambda x : 1 if  6 <=x <= 9 else 0 )
    # pickup hour 10-20  
    df_['hr_10_20'] = df_.pickup_hour.map(lambda x : 1 if  10 <=x <= 20 else 0  )
    # pickup hour 21-5    
    df_['hr_21_5'] = df_.pickup_hour.map(lambda x : 1 if  21 <=x <= 23 or 0 <= x <=5  else 0 )
    return df_
    
# data cleaning
def clean_data(df):
    df_ = df.copy()
    # remove potential distance outlier 
    df_ = df_[(df_['distance_haversine'] < df_['distance_haversine'].quantile(0.95))&
         (df_['distance_haversine'] > df_['distance_haversine'].quantile(0.05))]
    df_ = df_[(df_['distance_manhattan'] < df_['distance_manhattan'].quantile(0.95))&
         (df_['distance_manhattan'] > df_['distance_manhattan'].quantile(0.05))]
    # remove potential  trip duration outlier 
    # trip duration should less then 0.5 day and > 10 sec normally
    # in case test data has no trip duration 
    try:
        df_ = df_[(df_['trip_duration']  < 3*3600) & (df_['trip_duration'] > 10)]
        df_ = df_[(df_['trip_duration'] < df_['trip_duration'].quantile(0.95))&
             (df_['trip_duration'] > df_['trip_duration'].quantile(0.05))]
    # remove potential speed outlier  
        df_ = df_[(df_['avg_speed_h']  < 100) & (df_['avg_speed_h'] > 0)]
        df_ = df_[(df_['avg_speed_m']  < 100) & (df_['avg_speed_m'] > 0)]
        df_ = df_[(df_['avg_speed_h'] < df_['avg_speed_h'].quantile(0.95))&
         (df_['avg_speed_h'] > df_['avg_speed_h'].quantile(0.05))]
        df_ = df_[(df_['avg_speed_m'] < df_['avg_speed_m'].quantile(0.95))&
         (df_['avg_speed_m'] > df_['avg_speed_m'].quantile(0.05))]
    # remove the 2016-01-23 data since its too less comapre others days, 
    # maybe quality is not good 
        df_ = df_[(df_.pickup_date != '2016-01-23') &
                 (df_.dropoff_date != '2016-01-23')]
    except:
        pass
 
    # potential passenger_count outlier 
    df_ = df_[(df_['passenger_count']  <= 6) & (df_['passenger_count'] > 0)]
    
    return df_

def clean_data_(df):
    df_ = df.copy()
    # remove potential lon & lat  outlier 
    df_ = df_[(df_['pickup_latitude'] < df_['pickup_latitude'].quantile(0.99))&
         (df_['pickup_latitude'] > df_['pickup_latitude'].quantile(0.01))]
    df_ = df_[(df_['pickup_longitude'] < df_['pickup_longitude'].quantile(0.99))&
         (df_['pickup_longitude'] > df_['pickup_longitude'].quantile(0.01))]

    df_ = df_[(df_['dropoff_latitude'] < df_['dropoff_latitude'].quantile(0.99))&
         (df_['dropoff_latitude'] > df_['dropoff_latitude'].quantile(0.01))]
    df_ = df_[(df_['dropoff_longitude'] < df_['dropoff_longitude'].quantile(0.99))&
         (df_['dropoff_longitude'] > df_['dropoff_longitude'].quantile(0.01))]

    # remove the 2016-01-23 data since its too less comapre others days, 
    # maybe quality is not good 
    #df_ = df_[(df_.pickup_date != '2016-01-23') & (df_.dropoff_date != '2016-01-23')]
    # potential passenger_count outlier 
    df_ = df_[(df_['passenger_count']  <= 6) & (df_['passenger_count'] > 0)]
    
    return df_

def load_data():
    df_train = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/train.csv')
    df_test = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/test.csv')
    # sample train data for fast job 
    #df_train = df_train.sample(n=100)
    # clean train data 
    df_train_ = clean_data_(df_train)
    # merge train and test data for fast process and let model view test data when training as well 
    df_all = pd.concat([df_train_, df_test], axis=0)
    return df_all , df_train_ , df_test

def load_OSRM_data():
    # load OSRM dataset 
    # https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
    train_fastest_1 = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/fastest_routes_train_part_1.csv')
    train_fastest_2 = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/fastest_routes_train_part_2.csv')
    test_fastest = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/fastest_routes_test.csv')
    # merge 
    frame_fastest = pd.concat([train_fastest_1, train_fastest_2, test_fastest], axis = 0)
    # extract feature 
    right_turn = []
    left_turn = []
    right_turn+= list(map(lambda x:x.count('right')-x.count('slight right'),frame_fastest.step_direction))
    left_turn += list(map(lambda x:x.count('left')-x.count('slight left'),frame_fastest.step_direction))
    frame_fastest['right_turn'] = right_turn
    frame_fastest['left_turn'] = left_turn
    
    return frame_fastest
