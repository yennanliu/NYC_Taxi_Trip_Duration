
import java.io.*;
import java.sql.*;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;
import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;



/*

Credit 

1) https://github.com/jleetutorial/sparkTutorial/blob/master/src/main/java/com/sparkTutorial/sparkSql/HousePriceSolution.java


*/



public class data_preprocess {

  public static final String delimiter = ",";
  public static void main(String[] args) {
  // step 1 : read csv via CSVReader 
  //String csvFile = "train_data_java.csv";
  //CSVReader.read(csvFile);
  // step 2 : read csv via spark 
  //String args = "some arg";
  read_csv("test");

}



  public static void read_csv (String csvFile)

  {
    // PART 1  : load csv via spark session 
    SparkSession sparkSession = SparkSession.builder().appName("data_preprocess").config("spark.master", "local").getOrCreate();
    String PATH = "train_data_java.csv";
    Dataset<Row> rawData = sparkSession.read().option("header", "true").csv(PATH);
    Dataset<Row> transformedDataSet = rawData.withColumn("vendor_id", rawData.col("vendor_id").cast("double"))
    .withColumn("passenger_count", rawData.col("passenger_count").cast("double"))
    .withColumn("pickup_longitude", rawData.col("pickup_longitude").cast("double"))
    .withColumn("pickup_latitude", rawData.col("pickup_latitude").cast("double"))
    .withColumn("dropoff_longitude", rawData.col("dropoff_longitude").cast("double"))
    .withColumn("dropoff_latitude", rawData.col("dropoff_latitude").cast("double")) 
    .withColumn("trip_duration", rawData.col("trip_duration").cast("double")); 
    System.out.println(" ---------------- print csv : ----------------");
    // print first 20 row of csv 
    transformedDataSet.show(20);
    System.out.println(transformedDataSet);

    // PART 2  : aggregation via spark sql
    transformedDataSet.groupBy("vendor_id")
                        .agg(avg("pickup_longitude"), max("trip_duration"))
                        .show();


  }

}



