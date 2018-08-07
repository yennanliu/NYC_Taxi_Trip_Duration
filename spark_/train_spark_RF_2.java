
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.io.Serializable;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.AnalysisException;
import static org.apache.spark.sql.functions.col;


/*

Run the RF model again with pre-process data via 
1) SparkSQL
2) feature engineering from csv 

*/

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


public class train_spark_RF_2 {

    private static final String PATH = "train_data2_java.csv";

    public static void main(String[] args) {

        // initialise Spark session
        SparkSession sparkSession = SparkSession.builder().appName("train_spark_RF_2").config("spark.master", "local").getOrCreate();

        // load dataset, which has a header at the first row
        Dataset<Row> rawData = sparkSession.read().option("header", "true").csv(PATH);
            // -------------  SQL FOR EXTRACT FEATURES  -------------
        rawData.createOrReplaceTempView("df");
        Dataset<Row> sqlDF2 = sparkSession.sql("SELECT id, vendor_id, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_duration, month(pickup_datetime) as month ,year(pickup_datetime) as year , hour(pickup_datetime) as hour FROM df  ");
        sqlDF2.show();


        // cast the values of the features to doubles for usage in the feature column vector
        Dataset<Row> transformedDataSet = sqlDF2.withColumn("vendor_id", sqlDF2.col("vendor_id").cast("double"))
                .withColumn("passenger_count", sqlDF2.col("passenger_count").cast("double"))
                .withColumn("pickup_longitude", sqlDF2.col("pickup_longitude").cast("double"))
                .withColumn("pickup_latitude", sqlDF2.col("pickup_latitude").cast("double"))
                .withColumn("dropoff_longitude", sqlDF2.col("dropoff_longitude").cast("double"))
                .withColumn("dropoff_latitude", sqlDF2.col("dropoff_latitude").cast("double")) 
                .withColumn("trip_duration", sqlDF2.col("trip_duration").cast("double"))
                .withColumn("month", sqlDF2.col("month").cast("double"))
                .withColumn("year", sqlDF2.col("year").cast("double"))
                .withColumn("hour", sqlDF2.col("hour").cast("double")); 
                //.withColumn("dow", sqlDF2.col("dow").cast("double")); 

        // add a numerical label column for the Random Forest Classifier
        //transformedDataSet = transformedDataSet
        //        .withColumn("trip_duration", rawData.col("trip_duration").cast("double"));


        // identify the feature colunms
        String[] inputColumns = {"vendor_id","passenger_count", "pickup_longitude","pickup_latitude", "dropoff_longitude", "dropoff_latitude","month","year","hour"};
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