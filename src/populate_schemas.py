from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame




# Add the src/ directory to the Python path
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src/")))
#

try:
    from src.feature_extraction import processEpoch, processSub
    from src.schema_definition import get_feature_schema, get_subject_schema
    from src.preprocess_sets import subPath, participantsInfoPath, processSubPSDs, processSub
    from src.config_handler import load_config, initiate_config
except ImportError:
    from feature_extraction import processEpoch, processSub
    from schema_definition import get_feature_schema, get_subject_schema
    from preprocess_sets import subPath, participantsInfoPath, processSubPSDs, processSub
    from config_handler import load_config, initiate_config


try:
    config = load_config()
except RuntimeError:
    config = initiate_config()


def load_subjects_df(spark: SparkSession, participants_path: str="") -> DataFrame:
    """
    Reads participants.tsv and returns a Spark DataFrame
    with columns SubjectID and Group for groups A, C, and F.

    Parameters:
        spark (SparkSession): Active Spark session
        participants_path (str): Path to the participants.tsv file

    Returns:
        Spark DataFrame with SubjectID and Group columns
    """
    if len(participants_path) == 0:
        participantsInfo = pd.read_table(participantsInfoPath())
    else:
        participantsInfo = pd.read_table(participants_path)

    records = []
    for group_code in ["A", "C", "F"]:
        group_subjects = participantsInfo[participantsInfo["Group"] == group_code]["participant_id"].tolist()
        for sub in group_subjects:
            records.append((sub, group_code))
    return spark.createDataFrame(records, schema=get_subject_schema())




@pandas_udf(get_feature_schema(), PandasUDFType.GROUPED_MAP)
def extract_features_udtf(pdf):
    import time
    from feature_extraction import processEpoch, processSub
    from schema_definition import get_feature_schema, get_subject_schema
    rows = []
    start = time.time()
    for _, row in pdf.iterrows():
        subject_id = row["SubjectID"]
        try:
            print(f"Processing subject {subject_id}")
            epochs = processSub(subject_id, derivatives=False)
            print(f"Got {len(epochs)} epochs for {subject_id}")
            print(type(epochs)) 
            # for i, epoch in enumerate(epochs):
            for i in range(len(epochs)):
                epoch = epochs[i]
                if i < 2:  # Just print info for the first 2 epochs to avoid spam
                    # print(f"Epoch {i} shape: {epoch.to_data_frame().shape()}")
                    print(f"Epoch {i}")
                    print(type(epoch))
                    print(type(epochs[i]))
                epoch_id = f"ep-{i}"
                features = processEpoch(epoch)
                
                if i < 2:  # Debug output
                    # print(f"Epoch {i} features count: {len(features) if features else 0}")
                    if features and len(features) > 0:
                        pass
                        # print(f"First feature sample: {next(iter(features))}")
                
                for item in features:
                    # Check the structure of each item
                    # print(f"item {item}")
                    electrode_band_key, stats_value = item
                    electrode, band = electrode_band_key
                    # print(f"Adding: {subject_id}, {epoch_id}, {band}, {electrode}, stats: {stats_value}")
                    
                    # Add to results - adjust this based on actual structure 
                    try:
                        rows.append((subject_id, epoch_id, band, electrode, *stats_value))
                    except Exception as e:
                        print(f"Error appending row: {e}, stats_value: {stats_value}")
            
            print(f"Total rows collected: {len(rows)}")
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            
    # Print final row count before returning
    print(f"Returning DataFrame with {len(rows)} rows")
    
    # Check if we have column names from schema
    schema_fields = get_feature_schema()
    column_names = [f.name for f in schema_fields]
    print(f"Column names from schema: {column_names}")
    #error after here 
    import pandas as pd
    result_df = pd.DataFrame(rows, columns=column_names)
    # print(f"Result DataFrame shape: {result_df.shape}")
    print(time.time() - start)
    print(rows[0])
    print(schema_fields)
    return result_df


if __name__ == "__main__":
    import os
    import time
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
    import pandas as pd
    # sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src/")))
    #local imports
    from populate_schemas import load_subjects_df, extract_features_udtf
    from feature_extraction import processEpoch, processSub
    from schema_definition import get_feature_schema, get_subject_schema

    # Check if there's an active Spark context and stop it
    from pyspark import SparkContext
    if SparkContext._active_spark_context:
        print("Stopping existing Spark context...")
        SparkContext._active_spark_context.stop()
        print("Previous Spark context stopped successfully")
    
    # Set environment variables
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

    # Create new session with explicit local binding
    spark = SparkSession.builder \
        .appName("EEG_Analysis") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .master("local[*]") \
        .getOrCreate()

    print("New Spark session created successfully")

    spark = SparkSession.builder.appName("MyApp").getOrCreate()

    # preprocess_sets.setProjectRootDir()
    subject_df = load_subjects_df(spark)
    
    #we need the path of the src directory to add the local imports to our spark file
    # ROOT_DIR = ".."
    # SRC_DIR = ROOT_DIR + "/src"
    SRC_DIR = '.'

    sc = spark.sparkContext

    # Add all necessary modules to spark context
    try:
        sc.addPyFile(os.path.join(SRC_DIR, "feature_extraction.py"))
        print("Added feature_extraction.py to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "preprocess_sets.py"))
        print("Added preprocess_sets to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "schema_definition.py"))
        print("Added schema_definition.py to the pyspark context")
    except Exception as e:
        print(f"Error adding files to SparkContext: {e}")

    #Testing on a single subject 001
    result = (
    subject_df
    .filter((subject_df.SubjectID == "sub-001"))
    .groupBy("SubjectID")
    .apply(extract_features_udtf)
    )
    print("Subject 001 test and show results") 
    result.show()
    spark.stop()







