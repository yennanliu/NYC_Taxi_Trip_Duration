# python 3 
# -*- coding: utf-8 -*-



"""
* modify from 

    https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3178385260751176/1843063490960550/8430723048049957/latest.html

* spark ref 

    https://creativedata.atlassian.net/wiki/spaces/SAP/pages/83237142/Pyspark+-+Tutorial+based+on+Titanic+Dataset
    https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
    https://github.com/notthatbreezy/nyc-taxi-spark-ml/blob/master/python/generate-model.py


"""



# load basics library
import csv 
import os
import pandas as pd, numpy as np
import calendar
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
# spark 
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import count, avg
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler


# ---------------------------------
# config 
sc =SparkContext()
SparkContext.getOrCreate()
conf = SparkConf().setAppName("building a warehouse")
#sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)
print ("==================")
print (sc)
print ("==================")
# ---------------------------------






### ================================================ ###

# feature engineering 


# HELP FUNC 


### ================================================ ###


if __name__ == '__main__':
    # load data with spark way
    trainNYC = sc.textFile('train_data_java.csv').map(lambda line: line.split(","))
    headers = trainNYC.first()
    trainNYCdata = trainNYC.filter(lambda row: row != headers)
    sqlContext = SQLContext(sc)
    dataFrame = sqlContext.createDataFrame(trainNYCdata, ['id', 'vendor_id', 'passenger_count', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'trip_duration'])
    dataFrame.take(2)
    # convert string to float in  PYSPARK
    # https://stackoverflow.com/questions/46956026/how-to-convert-column-with-string-type-to-int-form-in-pyspark-data-frame
    dataFrame = dataFrame.withColumn("dropoff_longitude", dataFrame["dropoff_longitude"].cast("float"))
    dataFrame = dataFrame.withColumn("dropoff_latitude", dataFrame["dropoff_latitude"].cast("float"))
    inputFeatures = ["dropoff_longitude", "dropoff_latitude"]
    assembler = VectorAssembler(inputCols=inputFeatures, outputCol="features")
    output = assembler.transform(dataFrame)
    print (' ------- assembler.transform(dataFrame)------- '  )
    print (output.take(2))
    print (' ------- assembler.transform(dataFrame)------- '  )
    # pyspark RF model 
    model = RandomForest.trainRegressor(output, categoricalFeaturesInfo={},
                                    numTrees=50, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=3, maxBins=32)
    predictions = model.predict(test_data_points.map(lambda x: x.features))
    predicted_row_col = test_data_points.map(labeled_point_to_row_col_period).zip(predictions)

   #predicted_rasters = (trips_to_raster(predicted_row_col, 'predicted')
   #                  .map( lambda (period, (label, raster)): (period, raster)) )
   # observed_test_rasters = test_data.filter(lambda (x, d): 'observed' in d).mapValues(lambda d: d['observed'])
   # two_hour_avg = test_data.filter(lambda (x, d): 'two_hour' in d).mapValues(lambda d: d['two_hour']/2)















       
   


