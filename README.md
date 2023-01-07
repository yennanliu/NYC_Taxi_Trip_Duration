<p align="center"><img src ="https://github.com/yennanliu/NYC_Taxi_Trip_Duration/blob/master/data/nyc_taxi.jpg"></p>

<p align="center"><img src ="https://github.com/yennanliu/NYC_Taxi_Trip_Duration/blob/master/data/submit_log.png" ></p>

<h1 align="center"><a href="https://www.kaggle.com/c/nyc-taxi-trip-duration">NYC Taxi Trip Duration</a></h1>

<p align="center">
<!--- travis -->
<a href="https://travis-ci.org/yennanliu/NYC_Taxi_Trip_Duration"><img src="https://travis-ci.org/yennanliu/NYC_Taxi_Trip_Duration.svg?branch=master"></a>
<!--- coverage status -->
<a href='https://coveralls.io/github/yennanliu/NYC_Taxi_Trip_Duration?branch=master'><img src='https://coveralls.io/repos/github/yennanliu/NYC_Taxi_Trip_Duration/badge.svg?branch=master' alt='Coverage Status' /></a>
<!--- PR -->
<a href="https://github.com/yennanliu/NYC_Taxi_Trip_Duration/pulls"><img src="https://img.shields.io/badge/PRs-welcome-6574cd.svg"></a>
<!--- notebooks mybinder -->
<a href="https://mybinder.org/v2/gh/yennanliu/NYC_Taxi_Trip_Duration/master"><img src="https://img.shields.io/badge/launch-Jupyter-5eba00.svg"></a>
</p>

## INTRO

>Predict the total ride duration of taxi trips in New York City. primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

Please download the train data via : https://www.kaggle.com/c/nyc-taxi-trip-duration/data, and save at `data/train.csv`. Then 
you should be able to run the ML demo code (scripts under `run/`)

* [Kaggle Page](https://www.kaggle.com/c/nyc-taxi-trip-duration)
* [Analysis nb](https://nbviewer.jupyter.org/github/yennanliu/NYC_Taxi_Trip_Duration/blob/master/notebook/NYC_Taxi_EDA_V1_Yen.ipynb) - EDA ipython notebook 
* [ML nb](https://nbviewer.jupyter.org/github/yennanliu/NYC_Taxi_Trip_Duration/blob/master/notebook/NYC_Taxi_ML_V1_Yen.ipynb) - ML ipython notebook 
* [Main code](https://github.com/yennanliu/NYC_Taxi_Trip_Duration/tree/master/run) - Final ML code in python 
* [Team member repo](https://github.com/juifa-tsai/NYC_Taxi_Trip_Duration)

> Please also check [NYC_Taxi_Pipeline](https://github.com/yennanliu/NYC_Taxi_Pipeline) in case you are interested in the data engineering projects with similar taxi dataset. 

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
├── spark_   : Re-run the modeling with SPARK Mlib framework : JAVA / PYTHON / SCALA
└── start.sh : launch training env
```

## QUICK START

```Bash
# demo of  submit_xgb_387.py
cd NYC_Taxi_Trip_Duration
export PYTHONPATH=/Users/$USER/NYC_Taxi_Trip_Duration/
python run/submit_xgb_core_377_OSRM.py
....
KFold(n_splits=10, random_state=None, shuffle=False)
TRAIN: [ 134528  134529  134530 ... 1345276 1345277 1345278] TEST: [     0      1      2 ... 134525 134526 134527]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [134528 134529 134530 ... 269053 269054 269055]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [269056 269057 269058 ... 403581 403582 403583]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [403584 403585 403586 ... 538109 538110 538111]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [538112 538113 538114 ... 672637 672638 672639]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [672640 672641 672642 ... 807165 807166 807167]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [807168 807169 807170 ... 941693 941694 941695]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [ 941696  941697  941698 ... 1076221 1076222 1076223]
TRAIN: [      0       1       2 ... 1345276 1345277 1345278] TEST: [1076224 1076225 1076226 ... 1210749 1210750 1210751]
TRAIN: [      0       1       2 ... 1210749 1210750 1210751] TEST: [1210752 1210753 1210754 ... 1345276 1345277 1345278]
[0]	train-rmse:5.71905	valid-rmse:5.71921
# Multiple eval metrics have been passed: 'valid-rmse' will be used # for early stopping.

# Will train until valid-rmse hasn't improved in 250 rounds.

[15]	train-rmse:3.12115	valid-rmse:3.12236
[30]	train-rmse:1.7254	valid-rmse:1.72854
[45]	train-rmse:0.988145	valid-rmse:0.995816
[60]	train-rmse:0.61477	valid-rmse:0.63196
[75]	train-rmse:0.440141	valid-rmse:0.470721
[90]	train-rmse:0.365692	valid-rmse:0.408737
[105]	train-rmse:0.334895	valid-rmse:0.387246
[120]	train-rmse:0.320705	valid-rmse:0.379742
[135]	train-rmse:0.312278	valid-rmse:0.376631
[150]	train-rmse:0.306056	valid-rmse:0.375053
[165]	train-rmse:0.300808	valid-rmse:0.374119
[180]	train-rmse:0.296447	valid-rmse:0.373308
[195]	train-rmse:0.293421	valid-rmse:0.372817
[210]	train-rmse:0.290248	valid-rmse:0.372474
[225]	train-rmse:0.287413	valid-rmse:0.372054
[240]	train-rmse:0.285092	valid-rmse:0.37174
[255]	train-rmse:0.282846	valid-rmse:0.37154
[270]	train-rmse:0.28117	valid-rmse:0.371426
[285]	train-rmse:0.278903	valid-rmse:0.371133
[300]	train-rmse:0.276861	valid-rmse:0.370875
[315]	train-rmse:0.275129	valid-rmse:0.370712
[330]	train-rmse:0.273314	valid-rmse:0.370525
[345]	train-rmse:0.271538	valid-rmse:0.370385
[360]	train-rmse:0.270056	valid-rmse:0.370277
[375]	train-rmse:0.268084	valid-rmse:0.370096
[390]	train-rmse:0.266806	valid-rmse:0.370005
[405]	train-rmse:0.26524	valid-rmse:0.36985
[420]	train-rmse:0.26388	valid-rmse:0.369732
[435]	train-rmse:0.262156	valid-rmse:0.369577
[450]	train-rmse:0.260847	valid-rmse:0.369475
[465]	train-rmse:0.259224	valid-rmse:0.369379
[480]	train-rmse:0.257622	valid-rmse:0.369215
[495]	train-rmse:0.256402	valid-rmse:0.369111
[499]	train-rmse:0.256006	valid-rmse:0.369087
           trip_duration
id                      
id3004672     858.211670
id3505355     591.309570
id1217141     379.027527
id2150126     884.275757
id1598245     378.042847
id0668992     920.350037
id1765014    1413.732666
id0898117     968.557373
id3905224    3472.009277
id1543102     476.564056
id3024712    1047.087036
id3665810     330.427673
id1836461     452.128021
id3457080     630.039307
id3376065    1194.740723
id3008739     828.353821
id0902216    1145.874268
id3564824     519.362488

....

[625134 rows x 1 columns]

```

```Bash
# demo of submit_tpot.py 
cd NYC_Taxi_Trip_Duration
export PYTHONPATH=/Users/$USER/NYC_Taxi_Trip_Duration/
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

---
## PROCESS

```
EDA -> Data Preprocess -> Model select -> Feature engineering -> Model tune -> Prediction ensemble
```

```
# PROJECT WORKFLOW 

#### 1. DATA EXPLORATION (EDA)

Analysis : /notebook  

#### 2. FEATURE EXTRACTION 

2-1. **Feature dependency**
2-2. **Encode & Standardization** 
2-3. **Feature transformation** 
2-4. **Dimension reduction** ( via PCA) 

Script : /script 
Modeling : /run 

#### 3. PRE-TEST

3-1. **Input all standardized features to all models** <br>
3-2. **Regression**

#### 4. OPTIMIZATION

4-1. **Feature optimization** 
4-2. **Super-parameters tuning** 
4-3. **Aggregation**

#### 5. RESULTS 

-> check the output csv, log  
```
---

## Development 
```bash 
# unit test 
$ export PYTHONPATH=/Users/$USER/NYC_Taxi_Trip_Duration/
$ pytest -v tests/
# ============================== test session starts ==============================
# platform darwin -- Python 3.6.4, pytest-5.0.1, py-1.5.2, pluggy-0.12.0 -- /Users/jerryliu/anaconda3/envs/yen_dev/bin/python
# cachedir: .pytest_cache
# rootdir: /Users/jerryliu/NYC_Taxi_Trip_Duration
# plugins: cov-2.7.1
# collected 5 items                                                               

# tests/test_data_exist.py::test_training_data_exist PASSED                 [ 20%]
# tests/test_data_exist.py::test_validate_data_exist PASSED                 [ 40%]
# tests/test_udf.py::test_get_haversine_distance PASSED                     [ 60%]
# tests/test_udf.py::test_get_manhattan_distance PASSED                     [ 80%]
# tests/test_udf.py::test_get_direction PASSED                              [100%]

# =========================== 5 passed in 2.68 seconds ===========================
```

[![Star History Chart](https://api.star-history.com/svg?repos=yennanliu/NYC_Taxi_Trip_Duration&type=Date)](https://star-history.com/#yennanliu/NYC_Taxi_Trip_Duration&Date)

## REFERENCE

- XGBoost
  - http://xgboost.readthedocs.io/en/latest/python/python_api.html 
  - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
  - Core XGBoost Library VS scikit-learn API
  	- https://zz001.wordpress.com/2017/05/13/basics-of-xgboost-in-python/

- LightGBM
  - https://github.com/Microsoft/LightGBM/wiki/Installation-Guide
