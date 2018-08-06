/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//package org.apache.spark.examples.sql;


/*
credit 
https://github.com/apache/spark/blob/master/examples/src/main/java/org/apache/spark/examples/sql/JavaSparkSQLExample.java
*/

// $example on:programmatic_schema$
import java.util.ArrayList;
import java.util.List;
// $example off:programmatic_schema$
// $example on:create_ds$
import java.util.Arrays;
import java.util.Collections;
import java.io.Serializable;
// $example off:create_ds$

// $example on:schema_inferring$
// $example on:programmatic_schema$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
// $example off:programmatic_schema$
// $example on:create_ds$
import org.apache.spark.api.java.function.MapFunction;
// $example on:create_df$
// $example on:run_sql$
// $example on:programmatic_schema$
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off:programmatic_schema$
// $example off:create_df$
// $example off:run_sql$
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
// $example off:create_ds$
// $example off:schema_inferring$
import org.apache.spark.sql.RowFactory;
// $example on:init_session$
import org.apache.spark.sql.SparkSession;
// $example off:init_session$
// $example on:programmatic_schema$
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
// $example off:programmatic_schema$
import org.apache.spark.sql.AnalysisException;

// $example on:untyped_ops$
// col("...") is preferable to df.col("...")
import static org.apache.spark.sql.functions.col;
// $example off:untyped_ops$

public class spark_SQL {
  // $example on:create_ds$
  public static class Person implements Serializable {
    private String name;
    private int age;

    public String getName() {
      return name;
    }

    public void setName(String name) {
      this.name = name;
    }

    public int getAge() {
      return age;
    }

    public void setAge(int age) {
      this.age = age;
    }
  }
  // $example off:create_ds$

  public static void main(String[] args) throws AnalysisException {
    // $example on:init_session$
    SparkSession spark = SparkSession
      .builder()
      .appName("Java Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .config("spark.master", "local")
      .getOrCreate();
    // $example off:init_session$

    runBasicDataFrameExample(spark);
    //runDatasetCreationExample(spark);
    //runInferSchemaExample(spark);
    //runProgrammaticSchemaExample(spark);

    spark.stop();
  }

  private static void runBasicDataFrameExample(SparkSession spark) throws AnalysisException {
    // $example on:create_df$
    // keep csv header when read with spark 
    // https://stackoverflow.com/questions/27854919/how-do-i-skip-a-header-from-csv-files-in-spark
    Dataset<Row> df = spark.read().option("header","true").csv("train_data2_java.csv");
    
    // Displays the content of the DataFrame to stdout
    df.show();
    // +----+-------+
    // | age|   name|
    // +----+-------+
    // |null|Michael|
    // |  30|   Andy|
    // |  19| Justin|
    // +----+-------+
    // $example off:create_df$

    // $example on:untyped_ops$
    // Print the schema in a tree format
    df.printSchema();
    // root
    // |-- age: long (nullable = true)
    // |-- name: string (nullable = true)

    // Select only the "name" column
    df.select("id").show();

    // Select everybody, but increment the age by 1
    //df.select(col("id"), col("id").plus(1)).show();
    
    // Select pickup_latitude older than 40.76
    df.filter(col("pickup_latitude").gt(40.76)).show();
    // +---+----+
    // |age|name|
    // +---+----+
    // | 30|Andy|
    // +---+----+

    // Count trips by vendor_id
    df.groupBy("vendor_id").count().show();
    // +----+-----+
    // | age|count|
    // +----+-----+
    // |  19|    1|
    // |null|    1|
    // |  30|    1|
    // +----+-----+
    // $example off:untyped_ops$

    // $example on:run_sql$
    // Register the DataFrame as a SQL temporary view
    df.createOrReplaceTempView("df");
    Dataset<Row> sqlDF = spark.sql("SELECT * FROM df limit 10 ");
    sqlDF.show();

    // -------------  SQL FOR EXTRACT FEATURES  -------------
    df.createOrReplaceTempView("df");
    Dataset<Row> sqlDF2 = spark.sql("SELECT id, pickup_datetime, date(pickup_datetime) as date,month(pickup_datetime) as month,  date_format(pickup_datetime, 'EEEE')  as dow FROM df limit 10 ");
    sqlDF2.show();




    // $example on:global_temp_view$
    // Register the DataFrame as a global temporary view
    df.createGlobalTempView("df");

    // Global temporary view is tied to a system preserved database `global_temp`
    spark.sql("SELECT * FROM global_temp.df limit 10 ").show();
    // +----+-------+
    // | age|   name|
    // +----+-------+
    // |null|Michael|
    // |  30|   Andy|
    // |  19| Justin|
    // +----+-------+

    // Global temporary view is cross-session
    spark.newSession().sql("SELECT * FROM global_temp.df limit 10").show();
    // +----+-------+
    // | age|   name|
    // +----+-------+
    // |null|Michael|
    // |  30|   Andy|
    // |  19| Justin|
    // +----+-------+
    // $example off:global_temp_view$

    
  }

  
}