# -*- coding: utf-8 -*-
# load basics library 
import pandas as pd, numpy as np
import calendar
from sklearn.cluster import MiniBatchKMeans
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

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

def get_avg_travel_duration_speed(df):
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
    print (avg_speed)
    # merge 
    df_ = pd.merge(df_,avg_duration, how = 'left', on = ['pickup_weekday','pickup_hour'])
    df_ = pd.merge(df_,avg_speed, how = 'left', on = ['pickup_weekday','pickup_hour'])
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


if __name__ == '__main__':

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
    # get avg speed, duration
    df_all_ = get_avg_travel_duration_speed(df_all_)
    # label -> 0,1 
    df_all_ = label_2_binary(df_all_)
    # count trip over 60 min
    df_all_ = trip_over_60min(df_all_)
    # get log trip duration 
    df_all_['trip_duration_log'] = df_all_['trip_duration'].apply(np.log)
    # clean data 
    #df_all_ = clean_data(df_all_)
    print (df_all_.head())

    # modeling 
    features = [ 'vendor_id', 
                'passenger_count',
                'pickup_latitude', 
                'pickup_longitude', 
                'dropoff_latitude', 
                'dropoff_longitude',
                'pickup_pca0', 
                'pickup_pca1', 
                'dropoff_pca0', 
                'dropoff_pca1',
                'distance_haversine',
                'direction',
                'pca_manhattan',
                'pickup_time_delta',
                'pickup_month',
                'weekofyear', 
                'pickup_weekday', 
                'pickup_hour', 
                'week_delta',
                'week_delta_sin', 
                'pickup_hour_sin',
                'count_60min', 
                'dropoff_cluster_count']


    # split all data into train, test set 
    X_train = df_all_[df_all_['trip_duration'].notnull()][features].values
    y_train = df_all_[df_all_['trip_duration'].notnull()]['trip_duration_log'].values
    X_test  = df_all_[df_all_['trip_duration'].isnull()][features].values


### ================================================ ###
    
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor

    # train xgb  
    xgb = XGBRegressor(n_estimators=1000, max_depth=12, min_child_weight=150, 
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
        
        y_test += xgb.predict(X_test, ntree_limit=xgb.best_ntree_limit)
    y_test /= 2

### ================================================ ###

    df_sub = pd.DataFrame({
            'id': df_all[df_all['trip_duration'].isnull()]['id'].values,
             'trip_duration': np.exp(y_test)}).set_index('id')

    print (df_sub)
    df_sub.to_csv('~/NYC_Taxi_Trip_Duration/output/0824_xgb_387.csv')

    """
    #print (auto_classifier.config_dict)
    feature_importance = pd.DataFrame({'feature': features, \
                          'importance': xgboost_model.feature_importances_})\
                          .sort_values('importance',ascending =False)\
                          .set_index('feature')
    print (feature_importance)
     """

"""

    features = [ 'vendor_id', 
                'passenger_count',
                'pickup_latitude', 
                'pickup_longitude', 
                'dropoff_latitude', 
                'dropoff_longitude',
                'pickup_pca0', 
                'pickup_pca1', 
                'dropoff_pca0', 
                'dropoff_pca1',
                'distance_haversine',
                'direction',
                'pca_manhattan',
                'pickup_time_delta',
                'pickup_month',
                'weekofyear', 
                'pickup_weekday', 
                'pickup_hour', 
                'week_delta',
                'week_delta_sin', 
                'pickup_hour_sin',
                'count_60min', 
                'dropoff_cluster_count']



  xgb = XGBRegressor(n_estimators=1000, max_depth=12, min_child_weight=150, 
                   subsample=0.7, colsample_bytree=0.3)

# 0.38787

"""
