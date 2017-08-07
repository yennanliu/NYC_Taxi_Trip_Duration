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

## Quick Start

```
$export PYTHONPATH=/Users/yennanliu/NYC_Taxi_Trip_Duration/
$cd NYC_Taxi_Trip_Duration
$python run/submission.py

```



## Introduction

Predicts the total ride duration of taxi trips in New York City. primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

---
## Data & features extraction

### Data Exploration (EDA)
https://github.com/yennanliu/NYC_Taxi_Trip_Duration/tree/master/notebook

### Feature
1. **Overview**
   * Time & date dependency
   * Region/zone dependency    
   * **Wanted variables**
   ```
     * Id (label, 0x2)
     * Date-time: (28)
       * ~~Pickup weekday  (Label, 0x7)~~
       * ~~Pickup season   (label, 0x4)~~
       * ~~Pickup month    (label, 0x12)~~
       * ~~Pickup pm/am    (label, 0x2)~~  
       * ~~Pickup hrs      (label, 0x24)~~
     * Location (?+1)
       * ~~Pickup  zone    (label, 0x?)~~
       * ~~Dropoff zone    (label, 0x?)~~
       * ~~Pickup cb       (label, 0x?)~~
       * ~~Dropoff cb      (label, 0x?)~~
       * ~~Pickup neighborhoods (label, 0x?)~~
       * ~~Dropoff neighborhoods (label, 0x?)~~
       * Linear distance (float)
       * **Short Distance from Pickup to subway station** (float), see [CSV](../documents/NYC_Subway_Stations.csv)  
       * **Short Distance from Dropoff to subway station** (float), see [CSV](../documents/NYC_Subway_Stations.csv)
     * Passenger count   (int)
    ```
    * **Reference** <br />[New York City](https://en.wikipedia.org/wiki/Neighborhoods_in_New_York_City)<br />[List of Manhattan neighborhoods](https://en.wikipedia.org/wiki/List_of_Manhattan_neighborhoods)<br /> [List of Bronx neighborhoods](https://en.wikipedia.org/wiki/List_of_Bronx_neighborhoods)<br />[List of Brooklyn neighborhoods](https://en.wikipedia.org/wiki/List_of_Brooklyn_neighborhoods)<br />[List of Queens neighborhoods](https://en.wikipedia.org/wiki/List_of_Queens_neighborhoods)<br />[List of Staten Island neighborhoods](https://en.wikipedia.org/wiki/List_of_Staten_Island_neighborhoods)<br />[Subway Station coordinates](http://www.poi-factory.com/node/17432)
2. **Encode & Standardization**
3. **Feature transformation**
4. **Dimension reduction**
   * Use PCA
---
## Pre-test
Input all standardized features to all models.
### Model pre-performance
1. Regression
---
## Optimization
### Feature optimization
### Super-parameters tuning  
### Aggregation
---
## Results  
