# Databricks notebook source
# MAGIC %md ## Ingest csv file and create a Delta table

# COMMAND ----------

import os
from pyspark.sql.types import StringType, IntegerType, FloatType, StructField, StructType
from helpers import get_current_user

# COMMAND ----------

# MAGIC %md #### Create a personal workshop database

# COMMAND ----------

current_user = get_current_user()

spark.sql(f"CREATE DATABASE IF NOT EXISTS {current_user}")

spark.sql("USE {0}".format(current_user))

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %md #### Read in the csv file using schema inference

# COMMAND ----------

claim_attributes = (spark.read.format('csv')
                              .option("inferSchema", "true")
                              .option("header", "true")
                              .load(f"file:{os.getcwd()}/car_insurance_claim.csv")
                              .repartition(4))

display(claim_attributes)

# COMMAND ----------

# MAGIC %md #### View the inferred schema  

# COMMAND ----------

for column in claim_attributes.schema:
  print(column)

# COMMAND ----------

# MAGIC %md #### Specify the schema  
# MAGIC By specifying the schema we can be certain of the data types and also get better read performance for large files. Note that the Spark schema uses [complex types](https://spark.apache.org/docs/latest/sql-ref-datatypes.html).

# COMMAND ----------

integer_cols = ['ID', 'KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'TRAVTIME', 'TIF', 
                'CLM_FREQ', 'MVR_PTS', 'CAR_AGE', 'CLAIM_FLAG'] 

string_cols = ['BIRTH', 'INCOME', 'PARENT1', 'HOME_VAL', 'MSTATUS', 'GENDER',     
               'EDUCATION', 'OCCUPATION', 'CAR_USE', 'BLUEBOOK', 'CAR_TYPE',   
               'RED_CAR', 'OLDCLAIM', 'REVOKED', 'CLM_AMT', 'URBANICITY']

column_order = ['ID', 'KIDSDRIV', 'BIRTH','AGE','HOMEKIDS','YOJ','INCOME','PARENT1',
                'HOME_VAL','MSTATUS','GENDER','EDUCATION','OCCUPATION','TRAVTIME',
                'CAR_USE','BLUEBOOK','TIF','CAR_TYPE','RED_CAR','OLDCLAIM','CLM_FREQ',
                'REVOKED','MVR_PTS','CLM_AMT','CAR_AGE','CLAIM_FLAG','URBANICITY']

schema = StructType()
for column in column_order:
  if column in integer_cols:
    schema.add(StructField(column, IntegerType(), True))
    
  elif column in string_cols:
    schema.add(StructField(column, StringType(), True))
  
for col in schema:
  print(col)

# COMMAND ----------

claim_attributes = (spark.read.format('csv')
                                .schema(schema)
                                .option("header", "true")
                                .load(f"file:{os.getcwd()}/car_insurance_claim.csv")
                                .repartition(4))

display(claim_attributes)

# COMMAND ----------

# MAGIC %md ### Save the DataFrame as a distributed [Delta](https://docs.databricks.com/delta/delta-intro.html) table
# MAGIC Databricks Delta provides [enhanced performance and control](https://docs.databricks.com/delta/optimizations/index.html) of data lake data.

# COMMAND ----------

claim_attributes.write.mode('overwrite').format("delta").saveAsTable('claim_attributes_raw')

spark.table('claim_attributes_raw').count()

# COMMAND ----------

# MAGIC %md #### View the detailed table data by querying the Hive metastore  
# MAGIC  - Data files are stored in cloud object storage (S3)
# MAGIC  - Metadata about tables, including their object storage location, schema, partitions, etc. are store in the Hive metastore, which is backed by a relational database
# MAGIC  - Databricks interacts with the Hive metastore to capture metadata, then loads files from object storage, such as S3, that contain the actual data.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE TABLE EXTENDED claim_attributes_raw

# COMMAND ----------

# MAGIC %md #### View the files underlying our table  

# COMMAND ----------

dbutils.fs.ls('dbfs:/user/hive/warehouse/marshall_carter.db/claim_attributes_raw')
