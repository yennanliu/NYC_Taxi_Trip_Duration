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
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

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
    # pyspark KNN model 
    #Build a k-Means Clustering model
    kmeans = KMeans().setK(7).setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(100).setSeed(1)
    # Fit the model to training dataset
    model = kmeans.fit(output)
    print (' ------- KNN model output ------- '  )
    output_data = model.transform(output)
    print (output_data.take(30))
    #print (output_data.toDF().head(30))
    print (' ------- KNN model output ------- '  )
