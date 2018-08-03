// package nl.craftsmen.spark.iris;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

/****

# credit 

https://craftsmen.nl/an-introduction-to-machine-learning-with-apache-spark/
https://github.com/Silfen66/SparkIris/blob/master/src/main/java/nl/craftsmen/spark/iris/SparkIris.java

# ML spark feature vectore set up 

https://spark.apache.org/docs/latest/ml-classification-regression.html

**** /

/**

 * Apache Spark MLLib Java algorithm for classifying the Iris Species
 * into three categories using a Random Forest Classification algorithm.
 *

 **/


public class train_spark_RF {

    private static final String PATH = "train_data_java.csv";

    public static void main(String[] args) {

        // initialise Spark session
        SparkSession sparkSession = SparkSession.builder().appName("train_spark_RF").config("spark.master", "local").getOrCreate();

        // load dataset, which has a header at the first row
        Dataset<Row> rawData = sparkSession.read().option("header", "true").csv(PATH);

        // cast the values of the features to doubles for usage in the feature column vector
        Dataset<Row> transformedDataSet = rawData.withColumn("vendor_id", rawData.col("vendor_id").cast("double"))
                .withColumn("passenger_count", rawData.col("passenger_count").cast("double"))
                .withColumn("pickup_longitude", rawData.col("pickup_longitude").cast("double"))
                .withColumn("pickup_latitude", rawData.col("pickup_latitude").cast("double"))
                .withColumn("dropoff_longitude", rawData.col("dropoff_longitude").cast("double"))
                .withColumn("dropoff_latitude", rawData.col("dropoff_latitude").cast("double")) 
                .withColumn("trip_duration", rawData.col("trip_duration").cast("double")); 

        // add a numerical label column for the Random Forest Classifier
        //transformedDataSet = transformedDataSet
        //        .withColumn("trip_duration", rawData.col("trip_duration").cast("double"));


        // identify the feature colunms
        String[] inputColumns = {"vendor_id","passenger_count", "pickup_longitude","pickup_latitude", "dropoff_longitude", "dropoff_latitude"};
        VectorAssembler assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features");
        Dataset<Row> featureSet = assembler.transform(transformedDataSet);

        // split data random in trainingset (70%) and testset (30%) using a seed so results can be reproduced
        long seed = 5043;
        Dataset<Row>[] trainingAndTestSet = featureSet.randomSplit(new double[]{0.7, 0.3}, seed);
        Dataset<Row> trainingSet = trainingAndTestSet[0];
        Dataset<Row> testSet = trainingAndTestSet[1];

        trainingSet.show();

        // train the algorithm based on a Random Forest Classification Algorithm with default values
        RandomForestRegressor rf = new RandomForestRegressor()
                                    .setLabelCol("trip_duration")
                                    .setFeaturesCol("features");
        RandomForestRegressionModel rfModel = rf.fit(trainingSet);
        Dataset<Row> predictions = rfModel.transform(testSet);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                                        .setLabelCol("trip_duration")
                                        .setPredictionCol("prediction")
                                        .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        

        // test the model against the testset and show results
        System.out.println("----------------- prediction ----------------- ");
        predictions.select("id", "trip_duration", "prediction").show(20);
        System.out.println("----------------- prediction ----------------- ");

        // evaluate the model
        //RandomForestRegressionModel rfModel = (RandomForestRegressionModel)(model.stages()[1]);

        System.out.println("----------------- accuracy ----------------- "); 
        System.out.println("Trained RF model:\n" + rfModel.toDebugString());
        System.out.println("accuracy: " + rmse );
        System.out.println("----------------- accuracy ----------------- "); 
    }
}