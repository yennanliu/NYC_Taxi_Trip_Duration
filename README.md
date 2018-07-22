# NYC Taxi Trip Duration

[Kaggle Page](https://www.kaggle.com/c/nyc-taxi-trip-duration)<br>
[Forum](https://hackmd.io/s/BkScUQ4IW)



![image](https://github.com/yennanliu/NYC_Taxi_Trip_Duration/blob/master/data/nyc_taxi.jpg)

![image](https://github.com/yennanliu/NYC_Taxi_Trip_Duration/blob/master/data/submit_log.png)


## INTRO

Predicts the total ride duration of taxi trips in New York City. primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

## FILE STRUCTURE

```
├── README.md
├── data     : directory of train/test data 
├── documents: main reference 
├── model    : save the tuned model
├── notebook : main analysis
├── output   : prediction outcome
├── reference: other reference 
├── run      : fire the train/predict process 
├── script   : utility script for data preprocess / feature extract / train /predict  
├── spark_   : (dev)
└── start.sh : launch env



```


## QUICK START



```Bash
# demo of  submit_xgb_387.py
cd NYC_Taxi_Trip_Duration
export PYTHONPATH=/Users/youruserid/NYC_Taxi_Trip_Duration/
python run/submit_xgb_387.py

```

```Bash
# demo of submit_tpot.py 
cd NYC_Taxi_Trip_Duration
export PYTHONPATH=/Users/youruserid/NYC_Taxi_Trip_Duration/
python run/submit_tpot.py 

# output 

  dropoff_datetime  dropoff_latitude  dropoff_longitude         id  \
0 2016-03-14 17:32:30         40.765602         -73.964630  id2875421   
1 2016-06-12 00:54:38         40.731152         -73.999481  id2377394   
2 2016-01-19 12:10:48         40.710087         -74.005333  id3858529   
3 2016-01-30 22:09:03         40.749184         -73.992081  id0801584   
4 2016-06-17 22:40:40         40.765896         -73.957405  id1813257 
Best pipeline: ExtraTreesRegressor(input_matrix, bootstrap=True, max_features=0.8, min_samples_leaf=4, min_samples_split=8, n_estimators=100)
The cross-validation MSE
-0.1800694337127113
Imputing missing values in feature set
....

{'sklearn.linear_model.ElasticNetCV': {'tol': [1e-05, 0.0001, 0.001, 0.01, 0.1], 'l1_ratio': array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ])}, 'sklearn.cluster.FeatureAgglomeration': {'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], 'linkage': ['ward', 'complete', 'average']}, 'sklearn.preprocessing.Binarizer': {'threshold': array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ])}
.....

```


### Tech
- Python 3.4.5, Sklearn, Pandas 0.20.3 , Numpy, Xgboost, tpot


---
## PROCESS

```

EDA -> Data Preprocess -> Model select -> Feature engineering -> Model tune -> Prediction ensemble

```

```
#### 1. DATA EXPLORATION (EDA)

[Analysis](/notebook)

#### 2. FEATURE EXTRACTION 

[Script](/script)<br>
[Modeling](/run)

2-1. **Feature dependency**<br>
[variable](/reference/variable.md) <br>
2-2. **Encode & Standardization** <br>
2-3. **Feature transformation** <br>
2-4. **Dimension reduction**
   * Use PCA

#### 3. PRE-TEST
3-1. **Input all standardized features to all models** <br>
3-2. **Regression**

#### 4. OPTIMIZATION
4-1. **Feature optimization**<br>
4-2. **Super-parameters tuning** <br>
4-3. **Aggregation**<br>


#### 5. RESULTS  
```

---
## REFERENCE

- XGBoost
  - http://xgboost.readthedocs.io/en/latest/python/python_api.html 
  - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
  - Core XGBoost Library VS scikit-learn API
  	- https://zz001.wordpress.com/2017/05/13/basics-of-xgboost-in-python/

- LightGBM
  - https://github.com/Microsoft/LightGBM/wiki/Installation-Guide



