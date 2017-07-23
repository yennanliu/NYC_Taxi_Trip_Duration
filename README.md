# NYC Taxi Trip Duration
Kaggle web : https://www.kaggle.com/c/nyc-taxi-trip-duration

## File structure

```
├── README.md
├── data
├── notebook
│  
└── script
```

## Introduction
---
## Data & features extraction
### Train, validation & test data splitting
* train.csv - the training set (contains 1458644 trip records) : Will split to train set & validation set (30%)
* test.csv - the testing set (contains 625134 trip records) : Do not touch before validation has been done.
* sample_submission.csv - a sample submission file in the correct format
### Features
1. **Overview**
   * Time & date dependency?
   * Region/zone dependency?    
   * **Wanted variables**
     * Id (label, 0x2)
     * Date-time: (28)
       * Pickup weekday  (Label, 0x7)
       * Pickup season   (label, 0x4)
       * Pickup month    (label, 0x3)
       * Pickup pm/am    (label, 0x2)  
       * Pickup hrs      (label, 0x12)
     * Location (?+1)
       * Pickup  zone    (label, 0x?)
       * Dropoff zone    (label, 0x?)
       * Pickup cb       (label, 0x?)
       * Dropoff cb      (label, 0x?)
       * Pickup neighborhoods (label, 0x?)
       * Dropoff neighborhoods (label, 0x?)
       * Linear distance (float)
       * **Short Distance from Pickup to subway station** (float), see [CSV](../documents/NYC_Subway_Stations.csv)  
       * **Short Distance from Dropoff to subway station** (float), see [CSV](../documents/NYC_Subway_Stations.csv)
     * Passenger count   (int)
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
