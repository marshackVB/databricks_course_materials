# Databricks notebook source
# MAGIC %md ### Programming in Spark: options  
# MAGIC  - Koalas: The Pandas syntax ported to Spark  
# MAGIC  - Spark SQL syntax
# MAGIC  - Spark Dataframe syntax
# MAGIC  - A blend between Spark SQL and Dataframe syntax
# MAGIC  
# MAGIC  See [Spark API Reference](https://spark.apache.org/docs/latest/api/python/reference/index.html#api-reference)

# COMMAND ----------

import pyspark.sql.functions as func
from pyspark.sql.functions import when, col
from helpers import get_current_user

# COMMAND ----------

current_user = get_current_user()


spark.sql("USE {0}".format(current_user))

# COMMAND ----------

# MAGIC %md #### Koalas  
# MAGIC Koalas is a port of the Pandas syntax to Spark. It allows you to process data in parallel via Spark while leveraging the familiar Pandas API. There are some great tutorials and docs below where you can learn more.
# MAGIC  - [Databricks docs](https://docs.databricks.com/languages/koalas.html), including [sample notebook](https://docs.databricks.com/languages/koalas.html#pandas-to-koalas-notebook)
# MAGIC  - [Github repo](https://github.com/databricks/koalas)  
# MAGIC  - [Quick start](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_ps.html)

# COMMAND ----------

# MAGIC %md #### Spark SQL Syntax

# COMMAND ----------

# MAGIC %md Using %sql

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM claim_attributes_raw
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT CAR_TYPE,
# MAGIC        GENDER,
# MAGIC        COUNT(*) AS COUNT
# MAGIC FROM claim_attributes_raw
# MAGIC GROUP BY CAR_TYPE,
# MAGIC          GENDER
# MAGIC ORDER BY COUNT(*) DESC

# COMMAND ----------

# MAGIC %md #### Creating [temporary views](https://docs.databricks.com/spark/latest/spark-sql/language-manual/create-view.html) that can be accesed by other queries

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW view_example
# MAGIC AS SELECT ID,
# MAGIC           EDUCATION,
# MAGIC           AGE,
# MAGIC           rank() OVER (PARTITION BY EDUCATION ORDER BY AGE DESC) as RANK
# MAGIC    FROM claim_attributes_raw

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM view_example
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md #### The view can also be accessed by using the Spark DataFrame syntax

# COMMAND ----------

# Spark Dataframe syntax accessing the view

display(spark.table('view_example').limit(5))

# COMMAND ----------

# MAGIC %md #### For your SQL query development, also check out [Common Table Expressions (CTE)](https://docs.databricks.com/sql/language-manual/sql-ref-syntax-qry-select-cte.html).

# COMMAND ----------

# MAGIC %md #### Using the SQL API to return a DataFrame  
# MAGIC A benefit of this approach is that it's easy in parameterize SQL statements, include them in Python functions, and reference them in other DataFrame operations.

# COMMAND ----------

table = "claim_attributes_raw"
column = "AGE"
filter_condition = " <= 50"

df_filter = spark.sql(f"""SELECT *
                          FROM {table}
                          WHERE {column} {filter_condition}""")

display(df_filter)

# COMMAND ----------

# MAGIC %md Wrapping the above in a function

# COMMAND ----------

def sql_function_example(table, column, filter_condition):
  
  return spark.sql(f"""SELECT *
                       FROM {table}
                       WHERE {column} {filter_condition}""")

example_query = sql_function_example("claim_attributes_raw",  "AGE", " <= 50")

display(example_query)

# COMMAND ----------

# MAGIC %md #### Accessing a Spark Dataframe created using the DataFrame syntax in SQL
# MAGIC Create a temporary view based on the DataFrame

# COMMAND ----------

example_query.createOrReplaceTempView('example_query_table')

# COMMAND ----------

df_groupBy = spark.sql("""SELECT OCCUPATION,
                                 round(mean(CAR_AGE), 2) as MEAN_CAR_AGE
                                 
                          FROM example_query_table
                          GROUP BY OCCUPATION""")

display(df_groupBy)

# COMMAND ----------

# MAGIC %md ### Dataframe syntax  
# MAGIC  - [Quick start](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)  
# MAGIC  - [Dataframe API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html#dataframe-apis): searchable examples

# COMMAND ----------

# MAGIC %md #### SQL CASE statments three ways

# COMMAND ----------

df = spark.table("claim_attributes_raw")


"""
Option 1: Pure DataFrame operations
"""
higher_ed = ("PhD", "Master")
professional_occ = ("Doctor", "Professional")

transformed_df_1 = df.select(when(col("EDUCATION").isin(list(higher_ed)), 1).otherwise(0).alias("high_ed"),
                             when(col("OCCUPATION").isin(list(professional_occ)), 1).otherwise(0).alias("professional_occ"))


"""
Option 2: Pure Spark SQL
"""
df.createOrReplaceTempView("df_table")

transformed_df_2 = spark.sql(f"""SELECT CASE WHEN EDUCATION IN {higher_ed} THEN 1 ELSE 0 END AS high_ed,
                                        CASE WHEN OCCUPATION IN {professional_occ} THEN 1 ELSE 0 END AS professional_occ
                                 FROM df_table""")

"""
Option 3: A combinatino of Spark SQL and Datframe syntax using selectExpr() and a Python list of SQL expressions
"""
higher_ed = ("PhD", "Master")
professional_occ = ("Doctor", "Professional")


case_statements = [f"case when EDUCATION in {higher_ed} then 1 else 0 end as higher_ed",
                   f"case when OCCUPATION in {professional_occ} then 1 else 0 end as professional_occ"]

transformed_df_3 = df.selectExpr(case_statements)

display(transformed_df_3)

# COMMAND ----------

# MAGIC %md #### Common / useful DataFrame methods

# COMMAND ----------

# MAGIC %md selectExpr()

# COMMAND ----------

"""
selectExpr() will take a Python list of SQL commands. 
It's great when combined with Python list comprehension
"""        

transformation_cols = ["MSTATUS", "GENDER", "EDUCATION", "OCCUPATION", 
                       "HOME_VAL", "INCOME", "BLUEBOOK", "CLM_AMT", 
                       "URBANICITY", "CAR_TYPE", "OLDCLAIM"]

non_transformatoin_cols = [col for col in df.columns if col not in transformation_cols]

transformations = ["regexp_replace(MSTATUS, 'z_', '') as MSTATUS",
                   "regexp_replace(GENDER, 'z_', '') as GENDER",
                   "regexp_replace(EDUCATION, 'z_|<', '') as EDUCATION",
                   "regexp_replace(OCCUPATION, 'z_', '') as OCCUPATION",
                   "regexp_replace(URBANICITY, 'z_', '') as URBANICITY",
                   "regexp_replace(CAR_TYPE, 'z_', '') as CAR_TYPE",
                   
                   "cast(regexp_replace(HOME_VAL, '\\\$|,', '') as DOUBLE) as HOME_VAL",
                   "cast(regexp_replace(INCOME, '\\\$|,', '') as DOUBLE) as INCOME",
                   "cast(regexp_replace(BLUEBOOK, '\\\$|,', '') as DOUBLE) as BLUEBOOK",
                   "cast(regexp_replace(CLM_AMT, '\\\$|,', '') as DOUBLE) as CLM_AMT",
                   "cast(regexp_replace(OLDCLAIM, '\\\$|,', '') as DOUBLE) as OLDCLAIM"]


all_transformations = non_transformatoin_cols + transformations

"""
Methods can be chained together; the below is equivalent to a nest SQL query.
The results of the first selectExpr() statment are fed into the second
"""
df_transformed = (df.selectExpr(all_transformations))
                  
display(df_transformed)

# COMMAND ----------

# MAGIC %md A less repetitive version of the above leveraging Python list comprehension

# COMMAND ----------

# Regex expressions
trns_1 = 'z_'
trns_2 = 'z_|<'
trns_3 = '\\\$|,'


# Columns and transformations
trans_cols_1 = [('MSTATUS', trns_1), ('GENDER', trns_1), ('EDUCATION', trns_2), 
                ('OCCUPATION', trns_1), ('URBANICITY', trns_1), ('CAR_TYPE', trns_1)]
                   

trans_cols_2 = [('HOME_VAL', trns_3), ('INCOME', trns_3), ('BLUEBOOK', trns_3),
                ('CLM_AMT', trns_3), ('OLDCLAIM', trns_3)]


# Full transformation logic
trans_logic_1 = [f"regexp_replace({column}, '{transformation}', '') as {column}" 
                 for column, transformation in trans_cols_1]


trans_logic_2 = [f"cast(regexp_replace({column}, '{transformation}', '') as DOUBLE) as {column}" 
                 for column, transformation in trans_cols_2]


all_transformations = non_transformatoin_cols + trans_logic_1 + trans_logic_2

df_transformed = df.selectExpr(all_transformations)

for transformation in all_transformations:
  print(transformation)

# COMMAND ----------

display(df_transformed)

# COMMAND ----------

# MAGIC %md groupBy().agg()

# COMMAND ----------

decimals = 2

df_grouped_calcs = (df_transformed.groupBy('CAR_USE')
                                  .agg(func.round(func.sum('CLM_AMT'),  decimals).alias('TOTAL_CLM_AMT'),
                                       func.round(func.mean('AGE'),     decimals).alias('MEAN_AGE'),
                                       func.round(func.max('CLM_FREQ'), decimals).alias('MAX_CLM_FREQ'),
                                       func.count("*").alias('MEMBERS')))

# COMMAND ----------

"""Note that lazy evaluation is at play here. Nothing happens when the cell
above is executed. An action must be performed on the DataFrame, such as a show(), write(), 
count(), etc. command. This makes it possible to chain together many dataframe transformations 
and persist the final dataframe. Spark will utilize the Catalyst Optimizer to determine the most
efficient query plan to perform all the calculations
"""

display(df_grouped_calcs)

# COMMAND ----------

"""
We could have also used passed a Python list here, which opens up 
opportuniies for list comprehension
"""

grouped_calcs = [func.round(func.sum('CLM_AMT'),  decimals).alias('TOTAL_CLM_AMT'),
                 func.round(func.mean('AGE'),     decimals).alias('MEAN_AGE'),
                 func.round(func.max('CLM_FREQ'), decimals).alias('MAX_CLM_FREQ'),
                 func.count("*").alias('MEMBERS')]

df_grouped_calcs = (df_transformed.groupBy('CAR_USE')
                                  .agg(*grouped_calcs))

display(df_grouped_calcs)

# COMMAND ----------

# MAGIC %md Window()

# COMMAND ----------

from pyspark.sql import Window

# COMMAND ----------

row_number_window = Window.partitionBy('EDUCATION').orderBy(col('AGE').desc())

ranked_products = (df_transformed.select('EDUCATION', 'AGE')
                                 .withColumn('row_number', func.row_number().over(row_number_window)))

display(ranked_products)

# COMMAND ----------

# MAGIC %md join()
# MAGIC Additionally, joins using [where conditions](https://stackoverflow.com/questions/44966440/how-to-write-join-and-where-in-spark-dataframe-converting-sql-to-dataframe) 

# COMMAND ----------

# If columns don't have the same name
# join_to_self = df.join(df, col('ID') == col('ID'), how='inner')
join_to_self = df.join(df, ['ID'], how='inner')

display(join_to_self)

# COMMAND ----------

# MAGIC %md explode()  
# MAGIC This comes in handy for JSON and XML [parsing](https://docs.databricks.com/data/data-sources/read-json.html).

# COMMAND ----------

from pyspark.sql.types import ArrayType, IntegerType

# This is a UDF, we'll explore these in more detail in the UDF notebook
def create_range(size):
  return [i for i in range(size)]

create_range_udf = func.udf(create_range, ArrayType(IntegerType()))

num_rows = 10

range_df = df.withColumn("rows_array", func.lit(create_range_udf(func.lit(num_rows))))

display(range_df.select('ID', 'rows_array'))

# COMMAND ----------

# One row per array element
explode_df = (range_df.withColumn('exploded', func.explode(col('rows_array'))))

display(explode_df.select('ID', 'exploded', 'rows_array'))

# COMMAND ----------

# MAGIC %md Note that methods can be chained together and no computation occures until an action is called

# COMMAND ----------

grouped_calcs = [func.round(func.sum('CLM_AMT'),  decimals).alias('TOTAL_CLM_AMT'),
                 func.round(func.mean('AGE'),     decimals).alias('MEAN_AGE'),
                 func.round(func.max('CLM_FREQ'), decimals).alias('MAX_CLM_FREQ'),
                 func.count("*").alias('MEMBERS')]

# Cleaning the data then, performing the aggregations. Equivalent of a nested select
# in SQL. By legeraging Python and Spark DataFrame method chaining, you can accomplish
# alot with litte code.
df_mutiple_methods = (df.selectExpr(all_transformations)
                         .groupBy('EDUCATION')
                         .agg(*grouped_calcs)
                         .filter(col('MEAN_AGE') > 40))

# COMMAND ----------

# An action called on the DataFrame triggers the computation
display(df_mutiple_methods)

# COMMAND ----------

# Persist the cleaned up table
df_transformed.write.mode("overwrite").format("delta").saveAsTable("claim_attributes_transformed")
