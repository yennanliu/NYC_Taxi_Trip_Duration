# NYC Taxi Trip Duration

[Kaggle Page](https://www.kaggle.com/c/nyc-taxi-trip-duration)<br>
[Forum](https://hackmd.io/s/BkScUQ4IW)



![image](https://github.com/yennanliu/NYC_Taxi_Trip_Duration/blob/master/data/nyc_taxi.jpg)


## INTRO

Predicts the total ride duration of taxi trips in New York City. primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

## FILE STRUCTURE

```
├── README.md
├── data  
├── documents
├── model   
├── notebook
├── output
├── run
└── script


script : utility script for data prepare / modeling 
run    : fire the fitting process 
model  : save the tuned model
output : prediction outcome
notebook : main analysis

```


## QUICK START



```Bash
cd NYC_Taxi_Trip_Duration
export PYTHONPATH=/Users/youruserid/NYC_Taxi_Trip_Duration/
python run/submit_xgb_387.py

```

### Tech
- Python 3.4.5, Sklearn, Pandas 0.20.3 , Numpy, Xgboost, tpot


---
## PROCESS

```

EDA -> Data Preprocess -> Model select -> Feature engineering -> Model tune -> Prediction ensemble

```


#### 1. DATA EXPLORATION (EDA)

[Analysis](/notebook)

#### 2. FEATURE EXTRACTION 

[Script](/script)<br>
[Modeling](/run)

	-2-1. **Feature dependency**<br>
		[variable](/variable.md) <br>
	-2-2. **Encode & Standardization** <br>
	-2-3. **Feature transformation** <br>
	-2-4. **Dimension reduction**
   		* Use PCA

#### 3. PRE-TEST
	-3-1. **Input all standardized features to all models** <br>
	-3-2. **Regression**

#### 4. OPTIMIZATION
	-4-1. **Feature optimization**<br>
	-4-2. **Super-parameters tuning** <br>
	-4-3. **Aggregation**<br>

#### 5. RESULTS  

---
## REFERENCE

- xgboost
  - http://xgboost.readthedocs.io/en/latest/python/python_api.html 
  - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

- LightGBM
  - https://github.com/Microsoft/LightGBM/wiki/Installation-Guide



