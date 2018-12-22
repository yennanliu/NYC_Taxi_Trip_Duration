import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.*;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;

public class spark_SQL_analysis {

    private static final String PICKUP_LATITUDE_MIDPOINT = "pickup_latitude_midpoint";
    private static final String PASSENGER_COUNT_MIDPOINT = "passenger_count_midpoint";
    private static final String DROPOFF_LATITUDE_BUCKET = "dropoff_latitude_bucket";

    public static void main(String[] args) throws Exception {

        Logger.getLogger("org").setLevel(Level.ERROR);
        SparkSession session = SparkSession.builder().appName("NYC TAXI JAVA SPARK SQL ANALYSIS").master("local[1]").getOrCreate();

        DataFrameReader dataFrameReader = session.read();

        Dataset<Row> responses = dataFrameReader.option("header","true").csv("train_data2_java.csv");

        System.out.println("=== Print out schema ===");
        responses.printSchema();

        System.out.println("=== Print 20 records of responses table ===");
        responses.show(20);

        System.out.println("=== Print the pickup_datetime and dropoff_datetime columns of gender table ===");
        responses.select(col("pickup_datetime"),  col("dropoff_datetime")).show();

        System.out.println("=== Print records where the passenger_count is equal 3  ===");
        responses.filter(col("passenger_count").equalTo("3")).show();

        System.out.println("=== Print the count of passenger_count ===");
        RelationalGroupedDataset groupedDataset = responses.groupBy(col("passenger_count"));
        groupedDataset.count().show();

        System.out.println("=== Cast the PICKUP_LATITUDE_MIDPOINT and PASSENGER_COUNT_MIDPOINT to integer ===");
        Dataset<Row> castedResponse = responses.withColumn(PICKUP_LATITUDE_MIDPOINT, col("pickup_longitude").cast("integer"))
                                               .withColumn(PASSENGER_COUNT_MIDPOINT, col("passenger_count").cast("integer"));

        System.out.println("=== Print out casted schema ===");
        castedResponse.printSchema();

        System.out.println("=== Print records with PASSENGER_COUNT_MIDPOINT  less than 3 ===");
        castedResponse.filter(col(PICKUP_LATITUDE_MIDPOINT).$less(3)).show();

        System.out.println("=== Print the result by salary middle point in descending order ===");
        castedResponse.orderBy(col("pickup_latitude").desc()).show();

        System.out.println("=== Group by country and aggregate by average salary middle point and max age middle point ===");
        RelationalGroupedDataset datasetGroupByCountry = castedResponse.groupBy("vendor_id");
        datasetGroupByCountry.agg(avg("pickup_latitude"), max("pickup_latitude")).show();


        //Dataset<Row> responseWithSalaryBucket = castedResponse.withColumn(
        //        SALARY_MIDPOINT_BUCKET, col(DROPOFF_LATITUDE_BUCKET).divide(2).cast("integer").multiply(2));

        //System.out.println("=== With salary bucket column ===");
        //responseWithSalaryBucket.select(col(PICKUP_LATITUDE_MIDPOINT), col(SALARY_MIDPOINT_BUCKET)).show();

        //System.out.println("=== Group by salary bucket ===");
        //responseWithSalaryBucket.groupBy(SALARY_MIDPOINT_BUCKET).count().orderBy(col(SALARY_MIDPOINT_BUCKET)).show();

        session.stop();
    }
}
