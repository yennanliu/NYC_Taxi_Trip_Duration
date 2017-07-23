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
1. Overview
   * Time & date dependency?
   * Region/zone dependency?    
   * Wanted variables
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
       * Pickup block    (label, 0x?)
       * Dropoff block   (label, 0x?)
       * Linear distance (float)
     * Passenger count   (int)
2. Encode & Standardization
3. Feature transformation
4. Dimension reduction
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
