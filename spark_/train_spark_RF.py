# python 3 
# -*- coding: utf-8 -*-



######################################################################################################################################################
#  INTRO 
#
# * spark Mlib RandomForestRegressor 
#
#       https://stackoverflow.com/questions/33173094/random-forest-with-spark-get-predicted-values-and-r%C2%B2
#
# * modify from 
#
#       https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3178385260751176/1843063490960550/8430723048049957/latest.html
#
# * spark ref 
#
#       https://creativedata.atlassian.net/wiki/spaces/SAP/pages/83237142/Pyspark+-+Tutorial+based+on+Titanic+Dataset
#       https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/
#       https://github.com/notthatbreezy/nyc-taxi-spark-ml/blob/master/python/generate-model.py
#
#
#
######################################################################################################################################################



# load basics library
import csv 
import os
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
# spark 
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors


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
    dataFrame = dataFrame.withColumn("dropoff_longitude_", dataFrame["dropoff_longitude"].cast("float"))
    dataFrame = dataFrame.withColumn("dropoff_latitude_", dataFrame["dropoff_latitude"].cast("float"))
    dataFrame = dataFrame.withColumn("trip_duration_", dataFrame["trip_duration"].cast("float"))

    ############################## FIXED SOLUTION !!!! ##########################################################################################
    #
    #
    # https://stackoverflow.com/questions/46710934/pyspark-sql-utils-illegalargumentexception-ufield-features-does-not-exist
    # 
    # 
    ############################## FIXED SOLUTION !!!! ##########################################################################################

    dataFrame.registerTempTable("temp_sql_table")
    spark_sql_output=sqlContext.sql("""SELECT 
                        dropoff_longitude_,
                        dropoff_latitude_,
                        trip_duration_
                        FROM temp_sql_table """)
    print (spark_sql_output.take(10))

    trainingData=spark_sql_output.rdd.map(lambda x:(Vectors.dense(x[0:-1]), x[-1])).toDF(["features", "label"])
    trainingData.show()
    featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)

    (trainingData, testData) = trainingData.randomSplit([0.7, 0.3])
    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol="indexedFeatures")

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
    print ('='*100)
    print ('*** OUTCOME :')
    rmse = evaluator.evaluate(predictions)
    print(" *** Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    rfModel = model.stages[1]
    print(' *** : RF MODEL SUMMARY : ', rfModel)  # summary only
    print ('='*100)


