"""
PySpark + Great Expectations Data Quality Pipeline

Features:
- Schema enforcement
- Revenue calculation
- Great Expectations validation
- Data quality reporting
- Filtering invalid records
- Parquet output

Author: Your Name
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

import great_expectations as gx
from great_expectations.dataset import SparkDFDataset

# ----------------------------------------------------
# Logging Setup
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PySpark_GE_Pipeline")

# ----------------------------------------------------
# Spark Session
# ----------------------------------------------------
def create_spark():
    spark = (
        SparkSession.builder
        .appName("PySpark_GreatExpectations_Pipeline")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
def load_data(spark, path):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )

    df = df.withColumn("transaction_date", to_date(col("transaction_date")))
    return df

# ----------------------------------------------------
# Data Quality Validation
# ----------------------------------------------------
def validate_data(df):

    ge_df = SparkDFDataset(df)

    # Schema checks
    ge_df.expect_column_to_exist("transaction_id")
    ge_df.expect_column_to_exist("customer_id")
    ge_df.expect_column_to_exist("quantity")
    ge_df.expect_column_to_exist("price")
    ge_df.expect_column_to_exist("transaction_date")

    # Null checks
    ge_df.expect_column_values_to_not_be_null("transaction_id")
    ge_df.expect_column_values_to_not_be_null("customer_id")
    ge_df.expect_column_values_to_not_be_null("transaction_date")

    # Business rules
    ge_df.expect_column_values_to_be_between("quantity", min_value=1)
    ge_df.expect_column_values_to_be_between("price", min_value=0.01)

    # Uniqueness
    ge_df.expect_column_values_to_be_unique("transaction_id")

    results = ge_df.validate()

    return results

# ----------------------------------------------------
# Filter Valid Records
# ----------------------------------------------------
def filter_valid_records(df):
    return df.filter(
        (col("quantity") > 0) &
        (col("price") > 0) &
        (col("transaction_date").isNotNull())
    )

# ----------------------------------------------------
# Main Pipeline
# ----------------------------------------------------
def main():

    spark = create_spark()
    logger.info("Spark session started")

    input_path = "transactions.csv"
    output_path = "clean_output"

    df = load_data(spark, input_path)
    logger.info("Data loaded")

    validation_results = validate_data(df)

    logger.info("Validation Results:")
    logger.info(validation_results)

    if not validation_results["success"]:
        logger.warning("Data quality issues detected!")

    df_clean = filter_valid_records(df)

    df_clean = df_clean.withColumn(
        "revenue",
        col("quantity") * col("price")
    )

    df_clean.write.mode("overwrite").parquet(output_path)

    logger.info("Clean data written to Parquet")

    spark.stop()


if __name__ == "__main__":
    main()