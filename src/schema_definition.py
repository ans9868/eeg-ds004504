from pyspark.sql.types import *

# Subject table schema, *NOT ACTUALLY USED *
def get_subject_schema():
    return StructType([
        StructField("SubjectID", StringType(), False),
        StructField("Group", StringType(), False),
    ])

# Features table schema
def get_feature_schema():
    return StructType([
        StructField("SubjectID", StringType(), False),
        StructField("EpochID", StringType(), False),
        StructField("WaveBand", StringType(), False),
        StructField("Electrode", StringType(), False),
        StructField("Power", DoubleType(), True), # TODO: Change to DoubleType pro's and cons... is the rounding a big deal ? 
        StructField("Energy", DoubleType(), True) # TODO: Change to DoubleType pro's and cons... is the rounding a big deal ? 
        #     StructField("Skewness", FloatType(), True),
    #     StructField("Kurtosis", FloatType(), True),
    #     StructField("Variance", FloatType(), True),
    #     StructField("Min", FloatType(), True),
    #     StructField("Max", FloatType(), True),
    #     StructField("Mean", FloatType(), True)
    ])
