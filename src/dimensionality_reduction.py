from pyspark.sql.functions import col, concat_ws, lit
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import IntegerType


# from pyspark.sql.functions import when

# this z-scores all the data between 2 datasets - tested works 
def normalize_power(df_0, df_1):
    from pyspark.sql.functions import mean as _mean, stddev as _stddev, broadcast, when
    from pyspark.sql.types import FloatType

    df_0 = df_0.withColumn("Power", col("Power").cast(FloatType())) # cast as a 4 byte number 
    df_1 = df_1.withColumn("Power", col("Power").cast(FloatType()))

    stats = df_0.groupBy("Electrode", "WaveBand").agg(
        _mean("Power").alias("mean_power"),
        _stddev("Power").alias("std_power")
    )

    #2 more rows created mean and std power for each electrod and waveband
    df_0 = df_0.join(broadcast(stats), on=["Electrode", "WaveBand"]) 
    df_1 = df_1.join(broadcast(stats), on=["Electrode", "WaveBand"])

    # the .otherwise 1.0 makes it so that if the std power is 0 or null , then make it 1 , 
    # if std is 0 or null it means no varience in that variable (currnt_val == avg) and  so the result(s) in that column will be 0 as (currnet_val - mean) / 1 = 0
    df_0 = df_0.withColumn("Power", (col("Power") - col("mean_power")) / when((col("std_power").isNotNull()) & (col("std_power") != 0), col("std_power")).otherwise(1.0))
    df_1 = df_1.withColumn("Power", (col("Power") - col("mean_power")) / when((col("std_power").isNotNull()) & (col("std_power") != 0), col("std_power")).otherwise(1.0))

    # drop the mean and the std_power 
    df_0 = df_0.drop("mean_power", "std_power")
    df_1 = df_1.drop("mean_power", "std_power")
    return df_0, df_1



def prepare_features_for_pca(df):
    from pyspark.sql.functions import concat_ws

    df = df.withColumn("Electrode_WaveBand", concat_ws("_", "Electrode", "WaveBand"))
    pivot_keys = [row["Electrode_WaveBand"] for row in df.select("Electrode_WaveBand").distinct().collect()]

    features_df = (
        df.groupBy("SubjectID", "EpochID", "label")
        .pivot("Electrode_WaveBand", pivot_keys)
        .agg({"Power": "first"})
        .fillna(0.0)
    )
    feature_cols = [c for c in features_df.columns if c not in ("SubjectID", "EpochID", "label")]
    return features_df, feature_cols

def fit_pca_model(target_df, feature_cols, variance_target=0.95):
    from pyspark.ml.feature import PCA, VectorAssembler
    import numpy as np

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_train = assembler.transform(target_df)

    k_max = len(feature_cols)
    model_full = PCA(k=k_max, inputCol="features", outputCol="pca_features").fit(assembled_train)
    explained = model_full.explainedVariance.toArray()
    k_95 = next(i for i, x in enumerate(np.cumsum(explained)) if x >= variance_target) + 1

    pca_model = PCA(k=k_95, inputCol="features", outputCol="pca_features").fit(assembled_train)
    return pca_model, k_95


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

def apply_pca_model(target_df, pca_cols, pca_model, k):
    # Step 1: Assemble original features (before PCA) into a vector
    assembler = VectorAssembler(inputCols=pca_cols, outputCol="features")
    assembled_df = assembler.transform(target_df)

    # Step 2: Apply the PCA model
    transformed = pca_model.transform(assembled_df)

    # Step 3: Extract just the PCA output vector and label
    # Ensure label is IntegerType
    final_df = transformed.select(
        col("pca_features").alias("features"),
        col("label").cast(IntegerType()).alias("label")
    )

    return final_df






# def apply_pca_model(target_df, pca_cols, pca_model, k):
#     from pyspark.ml.feature import VectorAssembler
#     from pyspark.ml.functions import vector_to_array
#     from pyspark.sql.functions import col
#
#     assembler = VectorAssembler(inputCols=pca_cols, outputCol="features")
#     assembled_df = assembler.transform(target_df).select("features", "label")
#     
#
#     transformed = pca_model.transform(assembled_df).withColumn("pca_array", vector_to_array("pca_features"))
#     for i in range(k):
#         transformed = transformed.withColumn(f"PC{i+1}", col("pca_array")[i])
#     
#     assembled_df.withColumn("label", assembled_df["label"].cast(IntegerType()))
#
#     return transformed.select("label", *[f"PC{i+1}" for i in range(k)])
#












# def apply_pca(result_group_a, result_group_c, spark, variance_target=0.95):
#     import numpy as np
#     df_a, df_c = normalize_power(result_group_a, result_group_c)
#     df_a = df_a.withColumn("label", lit(0))
#     df_c = df_c.withColumn("label", lit(1))
#     full_df = df_a.union(df_c)
#
#     full_df = full_df.withColumn("Electrode_WaveBand", concat_ws("_", "Electrode", "WaveBand"))
#     pivot_keys = [row["Electrode_WaveBand"] for row in full_df.select("Electrode_WaveBand").distinct().collect()]
#
#     features_df = (
#         full_df.groupBy("SubjectID", "EpochID", "label")
#         .pivot("Electrode_WaveBand", pivot_keys)
#         .agg({"Power": "first"})
#         .fillna(0.0)
#     )
#
#     feature_cols = [c for c in features_df.columns if c not in ("SubjectID", "EpochID", "label")]
#     assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
#     assembled_df = assembler.transform(features_df).select("SubjectID", "EpochID", "label", "features")
#
#     k_max = len(feature_cols)
#     pca_model = PCA(k=k_max, inputCol="features", outputCol="pca_features").fit(assembled_df)
#     explained = pca_model.explainedVariance.toArray()
#     k_vals = next(i for i, x in enumerate(np.cumsum(explained)) if x >= variance_target) + 1
#
#     pca_model = PCA(k=k_vals, inputCol="features", outputCol="pca_features").fit(assembled_df)
#     pca_result = pca_model.transform(assembled_df)
#
#     pca_result = pca_result.withColumn("pca_array", vector_to_array("pca_features"))
#     for i in range(k_vals):
#         pca_result = pca_result.withColumn(f"PC{i+1}", col("pca_array")[i])
#
#     return pca_result.select("SubjectID", "EpochID", "label", *[f"PC{i+1}" for i in range(k_vals)])
#



