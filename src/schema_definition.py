from pyspark.sql.types import *

# Subject table schema, - not used but its good for reference -
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
        StructField("Electrode", StringType(), True),
        StructField("WaveBand", StringType(), True),
        StructField("FeatureName", StringType(), True), 
        StructField("FeatureValue", FloatType(), True), # can make into doubleType but doubletypes don't 'playnice' with most ML systems, decreasing accuracy significantly. *This was a tough lessong to learn*
        StructField("table_type", StringType(), True) # table type's are epoch, electrode and waveband in current setup
    ])
