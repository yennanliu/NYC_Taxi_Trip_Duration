# NYC Taxi Trip Duration
Kaggle page : https://www.kaggle.com/c/nyc-taxi-trip-duration
<br >Discussion Folum :point_right: <https://hackmd.io/s/BkScUQ4IW>

## File structure

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


## Introduction

Predicts the total ride duration of taxi trips in New York City. primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.


## Quick Start



```Bash
$export PYTHONPATH=/Users/yennanliu/NYC_Taxi_Trip_Duration/
$cd NYC_Taxi_Trip_Duration
$python run/submission.py

```

---
## PROCESS 

### 1. DATA EXPLORATION (EDA)
https://github.com/yennanliu/NYC_Taxi_Trip_Duration/tree/master/notebook

### 2. FEATURE EXTRACTION 

https://github.com/yennanliu/NYC_Taxi_Trip_Duration/tree/master/script
https://github.com/yennanliu/NYC_Taxi_Trip_Duration/tree/master/run

2-1. **Overview**
   * Time & date dependency
   * Region/zone dependency    
   ***Wanted variables**
   ```
     * Id (label, 0x2)
     * Date-time: (28)
       * Pickup year 
       * Pickup month 
       * Pickup weekday  (Label, 0x7)
       * Pickup hour
       * (todo)Pickup season   (label, 0x4)
       * Pickup month    (label, 0x12
       * (todo)Pickup pm/am    (label, 0x2)
     * Location (?+1)
       *  pickup_cluster
       *  dropoff_cluster
       * (todo)Pickup  zone    (label, 0x)
       * (todo)Dropoff zone    (label, 0x)
       * (todo)Pickup cb       (label, 0x?)
       * (todo)Dropoff cb      (label, 0x?)
       * (todo)Pickup neighborhoods (label, 0x)
       * (todo)Dropoff neighborhoods (label, 0x?)
       * (todo)Linear distance (float)
       * (todo)Short Distance from Pickup to subway station** (float), see [CSV](../documents/NYC_Subway_Stations.csv)  
       * (todo) Distance from Dropoff to subway station** (float), see [CSV](../documents/NYC_Subway_Stations.csv)
     * Passenger count   (int)
  ```
    **Reference** <br />[New York City](https://en.wikipedia.org/wiki/Neighborhoods_in_New_York_City)<br />[List of Manhattan neighborhoods](https://en.wikipedia.org/wiki/List_of_Manhattan_neighborhoods)<br /> [List of Bronx neighborhoods](https://en.wikipedia.org/wiki/List_of_Bronx_neighborhoods)<br />[List of Brooklyn neighborhoods](https://en.wikipedia.org/wiki/List_of_Brooklyn_neighborhoods)<br />[List of Queens neighborhoods](https://en.wikipedia.org/wiki/List_of_Queens_neighborhoods)<br />[List of Staten Island neighborhoods](https://en.wikipedia.org/wiki/List_of_Staten_Island_neighborhoods)<br />[Subway Station coordinates](http://www.poi-factory.com/node/17432)
   


2-2. **Encode & Standardization** <br>
2-3. **Feature transformation** <br>
2-4. **Dimension reduction**
   * Use PCA

### 3. PRE-TEST
3-1. **Input all standardized features to all models** <br>
3-2. **Regression**

### 4. OPTIMIZATION
4-1. **Feature optimization**<br>
4-2. **Super-parameters tuning** <br>
4-3. **Aggregation**<br>

### 5. RESULTS  
