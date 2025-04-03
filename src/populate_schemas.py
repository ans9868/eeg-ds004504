from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
from schema_definition import get_subject_schema, get_feature_schema
from feature_extraction import processEpoch, processSub
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
from feature_extraction import processEpoch, processSub
from schema_definition import get_feature_schema, get_subject_schema
from preprocess_sets import subPath, participantsInfoPath, processSubPSDs, processSub

def load_subjects_df(spark: SparkSession, participants_path: str) -> DataFrame:
    """
    Reads participants.tsv and returns a Spark DataFrame
    with columns SubjectID and Group for groups A, C, and F.

    Parameters:
        spark (SparkSession): Active Spark session
        participants_path (str): Path to the participants.tsv file

    Returns:
        Spark DataFrame with SubjectID and Group columns
    """
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
    from schema_definition import get_feature_schema
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
                    pass 
                    # print(f"Epoch {i}")
                    # print(type(epoch))
                    # print(type(epochs[i]))
                
                epoch_id = f'ep-{i}'
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
                        print(f"Error appending row: {e}, seubject_id: {subject_id}, epoch_id: {epoch_id}, band: {band}, electrode: {electrode}, stats_value: {stats_value}")
            
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
    return result_df


