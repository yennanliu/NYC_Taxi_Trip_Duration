### Utility scripts for trip duration prediction 


## Tech
- python 3, sklearn

## Scripts 

```
prepare.py :  scripts preparing data ready for training 
train.py   :  scripts train data 
run.py     :  execute the process (can be modified)

```

## Quick start 

```
$ python script/run.py 
# df_['pickup_week_'] = pd.to_datetime(df_.pickup_datetime,coerce=True).dt.weekday
#           id  vendor_id      pickup_datetime     dropoff_datetime  \
# 0  id2875421          2  2016-03-14 17:24:55  2016-03-14 17:32:30   
# 1  id2377394          1  2016-06-12 00:43:35  2016-06-12 00:54:38   ...
```






