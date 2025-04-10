from pyspark.sql.types import *

# Subject table schema, *NOT ACTUALLY USED *
def get_subject_schema():
    return StructType([
        StructField("SubjectID", StringType(), False),
        StructField("Group", StringType(), False),
    ])

# Features table schema !! swaped electrode and waveband!!
def get_feature_schema():
    return StructType([
        StructField("SubjectID", StringType(), False),
        StructField("EpochID", StringType(), False),
        StructField("Electrode", StringType(), False),
        StructField("WaveBand", StringType(), False),
        StructField("FeatureName", StringType(), True), 
        StructField("FeatureValue", DoubleType(), True),
        StructField("table_type", DoubleType(), True) 
        #     StructField("Skewness", FloatType(), True),
    #     StructField("Kurtosis", FloatType(), True),
    #     StructField("Variance", FloatType(), True),
    #     StructField("Min", FloatType(), True),
    #     StructField("Max", FloatType(), True),
    #     StructField("Mean", FloatType(), True)
    ])
