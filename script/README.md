### Utility scripts for trip duration prediction 


## Tech
- Python 3.4.5, Sklearn, Pandas 0.20.3 , Numpy, Xgboost, tpot

## Scripts 

```
prepare.py :  preparing data (feature extract / engineering /data cleaning...) 
train.py   :  training model 
model.py   :  model IO 

```

## Quick start 

```
# add script route to PYTHONPATH
$export PYTHONPATH=/Users/yennanliu/NYC_Taxi_Trip_Duration/

# to the project 
$cd NYC_Taxi_Trip_Duration

# fitting model with train data, save model as pickle file 
$python run/run_train.py

# predict test data with saved model, save prediction csv
$python run/run_test.py

```






