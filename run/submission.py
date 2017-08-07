# -*- coding: utf-8 -*-

# load basics library 
import pandas as pd, numpy as np
import calendar
from sklearn.cluster import MiniBatchKMeans
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split




### ================================================ ###

# feature engineering 


def get_time_feature(df):
    df_= df.copy()
    # pickup
    df_["pickup_date"] = pd.to_datetime(df_.pickup_datetime.apply(lambda x : x.split(" ")[0]))
    df_["pickup_hour"] = df_.pickup_datetime.apply(lambda x : x.split(" ")[1].split(":")[0])
    df_["pickup_year"] = df_.pickup_datetime.apply(lambda x : x.split(" ")[0].split("-")[0])
    df_["pickup_month"] = df_.pickup_datetime.apply(lambda x : x.split(" ")[0].split("-")[1])
    df_["pickup_weekday"] = df_.pickup_datetime.apply(lambda x :pd.to_datetime(x.split(" ")[0]).weekday())
    # weekday
    list(calendar.day_name)
    df_['pickup_week_'] = pd.to_datetime(df_.pickup_datetime,coerce=True).dt.weekday
    df_['pickup_weekday_'] = df_['pickup_week_'].apply(lambda x: calendar.day_name[x])
    # dropoff
    # in case test data dont have dropoff_datetime feature
    try:
        df_["dropoff_date"] = pd.to_datetime(df_.dropoff_datetime.apply(lambda x : x.split(" ")[0]))
        df_["dropoff_hour"] = df_.dropoff_datetime.apply(lambda x : x.split(" ")[1].split(":")[0])
        df_["dropoff_year"] = df_.dropoff_datetime.apply(lambda x : x.split(" ")[0].split("-")[0])
        df_["dropoff_month"] = df_.dropoff_datetime.apply(lambda x : x.split(" ")[0].split("-")[1])
        df_["dropoff_weekday"] = df_.dropoff_datetime.apply(lambda x :pd.to_datetime(x.split(" ")[0]).weekday())
    except:
        pass 
    return df_

# get time delta gap  
def get_time_feature2(df):
    df_ = df.copy()
    df_['pickup_datetime'] = pd.to_datetime(df_['pickup_datetime'])
    df_['pickup_time_delta'] = (df_['pickup_datetime'] - df_['pickup_datetime'].min()).dt.total_seconds()
    df_['week_delta'] = df_['pickup_datetime'].dt.weekday + \
                        ((df_['pickup_datetime'].dt.hour + \
                        (df_['pickup_datetime'].dt.minute / 60.0)) / 24.0)
    df_['weekofyear'] = df_['pickup_datetime'].dt.weekofyear
    return df_


# make weekday and hour cyclic, since we want to let machine understand 
# these features are in fact periodically 
def get_time_cyclic(df):
    df_ = df.copy()
    df_.pickup_hour = df_.pickup_hour.astype('int')
    df_['week_delta_sin'] = np.sin((df_['week_delta'] / 7) * np.pi)**2
    df_['pickup_hour_sin'] = np.sin((df_['pickup_hour'] / 24) * np.pi)**2
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


# PCA to transform longitude and latitude
# to improve decision tree performance 
from sklearn.decomposition import PCA
def pca_lon_lat(dftrain,dftest):
    X = np.vstack \
            ((dftrain[['pickup_latitude', 'pickup_longitude']].values,
              dftrain[['dropoff_latitude', 'dropoff_longitude']].values,
              dftest[['pickup_latitude', 'pickup_longitude']].values,
              dftest[['dropoff_latitude', 'dropoff_longitude']].values))
    # remove potential lon & lat outliers 
    min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
    max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
    X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]
    pca = PCA().fit(X)
    dftrain['pickup_pca0'] = pca.transform(dftrain[['pickup_latitude', 'pickup_longitude']])[:, 0]
    dftrain['pickup_pca1'] = pca.transform(dftrain[['pickup_latitude', 'pickup_longitude']])[:, 1]
    dftrain['dropoff_pca0'] = pca.transform(dftrain[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    dftrain['dropoff_pca1'] = pca.transform(dftrain[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    dftest['pickup_pca0'] = pca.transform(dftest[['pickup_latitude', 'pickup_longitude']])[:, 0]
    dftest['pickup_pca1'] = pca.transform(dftest[['pickup_latitude', 'pickup_longitude']])[:, 1]
    dftest['dropoff_pca0'] = pca.transform(dftest[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    dftest['dropoff_pca1'] = pca.transform(dftest[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    return dftrain,dftest 


# get lon & lat clustering for following avg location speed calculation
def get_clustering(df):
    coords = np.vstack((df_train[['pickup_latitude', 'pickup_longitude']].values,
                    df_train[['dropoff_latitude', 'dropoff_longitude']].values,
                    df_test[['pickup_latitude', 'pickup_longitude']].values,
                    df_test[['dropoff_latitude', 'dropoff_longitude']].values))
    df_ = df.copy()
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
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



### ======================== ###

def get_features(df):
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



### ================================================ ###

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








### ================================================ ###




def load_data():
	df_train = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/train.csv',nrows=50000)
	df_test = pd.read_csv('~/NYC_Taxi_Trip_Duration/data/test.csv')
	return df_train, df_test



### ================================================ ###


if __name__ == '__main__':

    df_train, df_test = load_data()
    #get basic features 
    df_train_ = get_time_feature(df_train)
    df_test_ = get_time_feature(df_test)
    df_train_ = get_time_feature2(df_train_)
    df_test_ = get_time_feature2(df_test_)
    df_train_ = get_time_cyclic(df_train_)
    df_test_ = get_time_cyclic(df_test_)
    # get other features 
    df_train_ = get_features(df_train_)
    df_test_ = get_features(df_test_)
    df_train_,df_test_ = pca_lon_lat(df_train_,df_test_)
    # get lon & lat clustering 
    df_train_= get_clustering(df_train_)
    df_test_= get_clustering(df_test_)
    # get avg ride count on dropoff cluster 
    df_train_ = trip_cluser_count(df_train_)
    df_test_ = trip_cluser_count(df_test_)
    # get log trip duration 
    df_train_['trip_duration_log'] = df_train_['trip_duration'].apply(np.log)
    # clean data 
    df_train_ = clean_data(df_train_)

    	# modeling 
    features = ['vendor_id',
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
    			 'distance_manhattan',
    			 'pickup_month',
    			 'pickup_hour',
    			 'week_delta_sin',
    			 'pickup_hour_sin',
    			 'dropoff_cluster_count']

    xtrain = df_train_.fillna(0)[features].values
    ytrain = df_train_['trip_duration_log'].values
    xtest = df_test_[features].values
    auto_classifier = TPOTRegressor(generations=2, population_size=8, verbosity=2)
    X_train, X_valid, y_train, y_valid = train_test_split(xtrain, ytrain,train_size=0.8, test_size=0.2)
    auto_classifier.fit(X_train, y_train)
    test_result = auto_classifier.predict(xtest)
    sub = pd.DataFrame()
    sub['id'] = df_test_['id']
    sub['trip_duration'] = np.exp(test_result)
    sub.to_csv('~/NYC_Taxi_Trip_Duration/output/Tpot_0803_submit.csv', index=False)
    sub.head()
















       
   
