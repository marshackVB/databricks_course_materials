# Databricks notebook source
# MAGIC %md # Delta compaction, optimization, and table operations
# MAGIC 
# MAGIC ###Technical deep dives:
# MAGIC  - See the list of Delta's optimization features [here](https://docs.databricks.com/delta/optimizations/index.html).  
# MAGIC  - Delta transaction log deep dive [workshop](https://www.youtube.com/watch?v=F91G4RoA8is&t=163s)   
# MAGIC  - Schema evolution deep file [workshop](https://www.youtube.com/watch?v=tjb10n5wVs8&list=RDCMUC3q8O3Bh2Le8Rj1-Q-_UUbA&index=2)  
# MAGIC  - Streaming data deep dive [workshop](https://www.youtube.com/watch?v=FePv0lro0z8&t=535s)  
# MAGIC  - Developing streaming pipelines [Tech session](https://www.youtube.com/watch?v=eOhAzjf__iQ&t=721s)
# MAGIC  - Unpacking the transaction log [blog](https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html)
# MAGIC  - Top tuning tips [tech session](https://www.youtube.com/watch?v=hcoMHnTcvmg)

# COMMAND ----------

import numpy as np
import pandas as pd

import pyspark.sql.functions as func
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField, StringType
from helpers import get_current_user, get_num_files, list_files, bytes_to_megabytes

# COMMAND ----------

current_user = get_current_user()

spark.sql("USE {0}".format(current_user))

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM claim_attributes_transformed
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md #### Get table metadata, including location of files

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE EXTENDED claim_attributes_transformed

# COMMAND ----------

# MAGIC %md #### View underlying table files

# COMMAND ----------

data_dir = 'dbfs:/user/hive/warehouse/marshall_carter.db/claim_attributes_transformed'

list_files(data_dir)

# COMMAND ----------

# MAGIC %md ####Get a count of files in a directory

# COMMAND ----------

get_num_files(data_dir)

# COMMAND ----------

# MAGIC %md ## The many small files problem
# MAGIC A deeper discuss of this problem is available [here](https://databricks.com/blog/2018/07/31/processing-petabytes-of-data-in-seconds-with-databricks-delta.html). Let's create a table version where the underlying files are small, thus resulting in many small files.

# COMMAND ----------

# MAGIC %md #### Generate a large version of the table by duplicating it many times

# COMMAND ----------

spark_df = spark.table('claim_attributes_transformed')

# Create a UDF that will generate an array of integers
def create_range(size):
  return [i for i in range(size)]

create_range_udf = func.udf(create_range, ArrayType(IntegerType()))

# Number of duplicates
num_rows = 1000

# Generate a much larger table by creating duplicate rows
range_df = (spark_df.withColumn("rows_array", func.lit(create_range_udf(func.lit(num_rows))))
                    .withColumn('exploded', func.explode(col('rows_array')))
                    .withColumn('DISTINCT_ID', func.concat(col('ID'), func.lit('_'), col('exploded'))) # Not necessary?
                    .drop('rows_array') 
                    # Spread the data across many small Spark partitions, which will
                    # result in many small files once the table is written if using 
                    # the traditional Parquet format
                    .repartition(2000))
          
range_df_temptable = range_df.createOrReplaceTempView('range_df_view')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS claim_attributes_many_files;
# MAGIC 
# MAGIC CREATE TABLE claim_attributes_many_files
# MAGIC USING DELTA
# MAGIC AS SELECT * FROM range_df_view;

# COMMAND ----------

spark.table('claim_attributes_many_files').count()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE TABLE EXTENDED claim_attributes_many_files

# COMMAND ----------

table_dir = "dbfs:/user/hive/warehouse/marshall_carter.db/claim_attributes_many_files"

list_files(table_dir)

# COMMAND ----------

bytes_to_megabytes(246568)

# COMMAND ----------

get_num_files(table_dir)

# COMMAND ----------

# MAGIC %md #### Notice how long this query takes to complete

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT CAR_TYPE,
# MAGIC        COUNT(*) as COUNT
# MAGIC 
# MAGIC FROM claim_attributes_many_files
# MAGIC 
# MAGIC GROUP BY CAR_TYPE

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC DROP TABLE IF EXISTS claim_attributes_many_files

# COMMAND ----------

# MAGIC %md #### Compact the small files using Delta's [Auto Optimize](https://docs.databricks.com/delta/optimizations/auto-optimize.html) functionality.
# MAGIC We can leverage "autoOptimize.optimizeWrite" to compact these small files into large ones before being written to the Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS claim_attributes_optimized;
# MAGIC 
# MAGIC CREATE TABLE claim_attributes_optimized
# MAGIC USING DELTA
# MAGIC TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true)
# MAGIC AS SELECT * FROM range_df_view;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE TABLE EXTENDED claim_attributes_optimized

# COMMAND ----------

# MAGIC %md Many fewer, larger files underly the table now

# COMMAND ----------

table_dir = "dbfs:/user/hive/warehouse/marshall_carter.db/claim_attributes_optimized"

list_files(table_dir)

# COMMAND ----------

bytes_to_megabytes(101525468)

# COMMAND ----------

# MAGIC %md #### Notice how much faster the query runs  
# MAGIC %md The query is now much faster. Subsequent queries could be even faster than this one if leveraging [Delta caching](https://docs.databricks.com/delta/optimizations/delta-cache.html).

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT CAR_TYPE,
# MAGIC        COUNT(*) as COUNT
# MAGIC FROM claim_attributes_optimized
# MAGIC GROUP BY CAR_TYPE

# COMMAND ----------

# MAGIC %md #### What about small files accumulation over time?  
# MAGIC In a real-world scenario, small files could accumulate in a table over time. Lets simulate this by inserting more small files into the table.  
# MAGIC NOTE: This takes several minutes to complete

# COMMAND ----------

sample = spark.table("claim_attributes_optimized").limit(1000)
sample.createOrReplaceTempView('sample_temp_table')
sample.cache()

for i in range(50):
  
  spark.sql("INSERT INTO claim_attributes_optimized SELECT * FROM sample_temp_table")

# COMMAND ----------

# MAGIC %md There are now more, small files in the directory, but this isn't telling the entire story.

# COMMAND ----------

dir = 'dbfs:/user/hive/warehouse/marshall_carter.db/claim_attributes_optimized'

get_num_files(dir)

# COMMAND ----------

list_files(dir)

# COMMAND ----------

# MAGIC %md #### View the table's history using [time travel](https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html).
# MAGIC Scrolling down the operations in the table's history, we find the operation, "OPTIMIZE", where "auto":"true". This automatically compacts small files periodically after a certain number of them have accumulated in the table. The small files were retained in the directory to enable time travel functionality.

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC DESCRIBE HISTORY claim_attributes_optimized;

# COMMAND ----------

# MAGIC %md #### Time travel allows you to view the table as it existing in the past

# COMMAND ----------

# MAGIC %sql  
# MAGIC 
# MAGIC SELECT count(*)
# MAGIC FROM claim_attributes_optimized
# MAGIC VERSION AS OF 0

# COMMAND ----------

# MAGIC %sql  
# MAGIC 
# MAGIC SELECT count(*)
# MAGIC FROM claim_attributes_optimized
# MAGIC VERSION AS OF 51

# COMMAND ----------

# MAGIC %md #### Remove all time travel's historical data snapshots using [VACUUM](https://docs.databricks.com/delta/delta-utility.html#vacuum)
# MAGIC Note that this will prevent us from restoring a prior version of the table; we are doing this purely for example purposes.

# COMMAND ----------

# Turning this setting off is required to remove all historical files
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC VACUUM claim_attributes_optimized RETAIN 0 HOURS;

# COMMAND ----------

# MAGIC %md Notice that the smaller files were automatically compacted into larger ones.

# COMMAND ----------

dir = "dbfs:/user/hive/warehouse/workshop_mlc.db/claim_attributes_optimized"

get_num_files(dir)

# COMMAND ----------

list_files(dir)

# COMMAND ----------

# MAGIC %md #### Delta table DDL operations

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS claim_attributes_ddl;
# MAGIC 
# MAGIC CREATE TABLE claim_attributes_ddl
# MAGIC USING DELTA
# MAGIC TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true)
# MAGIC AS SELECT ID, 
# MAGIC           AGE,
# MAGIC           CAR_TYPE 
# MAGIC FROM claim_attributes_transformed

# COMMAND ----------

display(spark.table('claim_attributes_ddl'))

# COMMAND ----------

# MAGIC %md Delete

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC DELETE FROM claim_attributes_ddl WHERE ID = 710125256

# COMMAND ----------

# MAGIC %md Update

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC UPDATE claim_attributes_ddl SET `AGE` = 100 WHERE ID = 258417857

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT ID, AGE
# MAGIC FROM claim_attributes_ddl
# MAGIC WHERE ID = 258417857
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md Merge into  
# MAGIC Check out [SCD Type 2 Merges](https://docs.databricks.com/_static/notebooks/merge-in-scd-type-2.html) in Databricks

# COMMAND ----------

raw_update = [(728563911,  70, 'Sportscar'), 
              (96350592,   80, 'Sportscar'),
              (490562058,  90, 'Sportscar'),
              (999,       100, 'Sprotscar')]

schema = StructType([StructField('ID',       IntegerType(), True),
                     StructField('AGE',      IntegerType(), True),
                     StructField('CAR_TYPE', StringType(),  True)])

update_df = spark.createDataFrame(raw_update, schema)
update_df.createOrReplaceTempView('updated_df_table')

display(update_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO  claim_attributes_ddl as d
# MAGIC USING updated_df_table as m
# MAGIC on d.ID = m.ID
# MAGIC WHEN MATCHED THEN 
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED 
# MAGIC   THEN INSERT *

# COMMAND ----------

existing_claims = [728563911, 96350592, 490562058]
new_claims = [999]

display(spark.table('claim_attributes_ddl').filter(col('ID').isin(existing_claims + new_claims)))

# COMMAND ----------

# MAGIC %md Leverage Delta Time Travel to view these changes

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC DESCRIBE HISTORY claim_attributes_ddl;

# COMMAND ----------

# MAGIC %md View tables at different states

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC SELECT *
# MAGIC FROM claim_attributes_ddl
# MAGIC VERSION AS OF 0) s
# MAGIC WHERE ID IN (728563911, 96350592, 490562058, 999);

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC SELECT *
# MAGIC FROM claim_attributes_ddl
# MAGIC VERSION AS OF 3) s
# MAGIC WHERE ID IN (728563911, 96350592, 490562058, 999);

# COMMAND ----------

# MAGIC %md Schema evolution

# COMMAND ----------

raw_update = [(8888, 70, 'Sportscar', 'M')]

schema = StructType([StructField('ID',       IntegerType(),  True),
                     StructField('AGE',      IntegerType(),  True),
                     StructField('CAR_TYPE', StringType(),   True),
                     StructField('GENDER',   StringType(),   True)])

update_df = spark.createDataFrame(raw_update, schema)

display(update_df)

# COMMAND ----------

# MAGIC %md Schema enforcement prevents this update

# COMMAND ----------

update_df.write.format("delta").mode("append").saveAsTable('claim_attributes_ddl')

# COMMAND ----------

# MAGIC %md Allow schema merge

# COMMAND ----------

update_df.write.option("mergeSchema","true").format("delta").mode("append").saveAsTable('claim_attributes_ddl')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM claim_attributes_ddl
# MAGIC ORDER BY GENDER DESC
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md Cleanup

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %md Cleanup

# COMMAND ----------

tables = ['claim_attributes_optimized', 'claim_attributes_many_files', 'claim_attributes_ddl', 
          'claim_attributes_raw', 'claim_attributes_transformed']

for table in tables:
  spark.sql(f"DROP TABLE IF EXISTS {table}")
  
spark.sql("DROP DATABASE IF EXISTS {0}".format(current_user))
