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
        StructField("Electrode", StringType(), True),
        StructField("WaveBand", StringType(), True),
        StructField("FeatureName", StringType(), True), 
        StructField("FeatureValue", FloatType(), True), # can make into doubleType for more accuracy but it is bad for ML
        StructField("table_type", StringType(), True) 
        #     StructField("Skewness", FloatType(), True),
    #     StructField("Kurtosis", FloatType(), True),
    #     StructField("Variance", FloatType(), True),
    #     StructField("Min", FloatType(), True),
    #     StructField("Max", FloatType(), True),
    #     StructField("Mean", FloatType(), True)
    ])
