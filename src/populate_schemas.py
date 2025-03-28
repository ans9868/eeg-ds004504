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


# to test
@pandas_udf(get_feature_schema(), PandasUDFType.GROUPED_MAP)
def extract_features_udtf(pdf):
    rows = []
    for _, row in pdf.iterrows():
        subject_id = row["SubjectID"]
        try:
            epochs = processSub(subject_id, derivatives=False)
            for i, epoch in enumerate(epochs):
                epoch_id = f"{subject_id}_ep{i}"
                features = processEpoch(epoch)
                for (electrode, band), stats in features:
                    # Assuming `stats` is a tuple with (mean, variance, skewness, kurtosis)
                    rows.append((subject_id, epoch_id, band, electrode, *stats))
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    return pd.DataFrame(rows, columns=[f.name for f in get_feature_schema()])

# Tested
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

    return spark.createDataFrame(records, schema=get_subject_schema()) # subject schmea is 'SubjectID', 'Group'




# @pandas_udf(get_feature_schema(), PandasUDFType.GROUPED_MAP)
# def extract_features_udtf(pdf):
#     rows = []
#     for _, row in pdf.iterrows():
#         subject_id = row["SubjectID"]
#         try:
#             epochs = processSub(subject_id, derivatives=False)
#             for i, epoch in enumerate(epochs):
#                 epoch_id = f"{subject_id}_ep{i}"
#                 features = processEpoch(epoch)
#                 for (electrode, band), stats in features:
#                     # Assuming `stats` is a tuple with (mean, variance, skewness, kurtosis)
#                     rows.append((subject_id, epoch_id, band, electrode, *stats))
#         except Exception as e:
#             print(f"Error processing {subject_id}: {e}")
#     return pd.DataFrame(rows, columns=[f.name for f in get_feature_schema()])
#
#
# # Tested
# @pandas_udf(get_subject_schema(), PandasUDFType.GROUPED_MAP)
# def generate_subjects_udtf(pdf):
#     # Read the participants.tsv file
#     participantsInfo = pd.read_table('../ds004504/participants.tsv')
#     
#     # Filter the subjects by group and prepare the data
#     subjects = []
#     for group, group_name in [("A", "GroupA"), ("C", "GroupC"), ("F", "GroupD")]:
#         group_subjects = participantsInfo[participantsInfo["Group"] == group]["participant_id"].tolist()
#         subjects.extend([{"SubjectID": sub, "Group": group_name} for sub in group_subjects])
#     
#     # Return a DataFrame with the subjects and their groups
#     return pd.DataFrame(subjects)
