from pyspark.sql.functions import col, concat_ws, lit
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import IntegerType


# TODO: make config for variance target 


from pyspark.sql.functions import col, min as _min, max as _max, broadcast, when

def min_max_normalize(train_df, test_df, feature_cols, target_min=-1.0, target_max=1.0):
    # 1. Compute min and max for each feature on the training set only
    stats_exprs = [
        _min(c).alias(f"{c}_min") for c in feature_cols
    ] + [
        _max(c).alias(f"{c}_max") for c in feature_cols
    ]
    stats = train_df.agg(*stats_exprs).collect()[0]
    
    # 2. Normalize both train and test using train stats
    def apply_minmax(df):
        for c in feature_cols:
            col_min = float(stats[f"{c}_min"])
            col_max = float(stats[f"{c}_max"])
            range_val = col_max - col_min if col_max != col_min else 1.0  # avoid div by 0

            df = df.withColumn(
                c,
                ((col(c) - col_min) / range_val) * (target_max - target_min) + target_min
            )
        return df

    return apply_minmax(train_df), apply_minmax(test_df)



def normalize_by_column(train_df, test_df, feature_cols):
    from pyspark.sql.functions import col, mean as _mean, stddev as _stddev, broadcast, when, first
    from pyspark.sql.types import FloatType

    # Cast all feature columns to float
    for col_name in feature_cols:
        train_df = train_df.withColumn(col_name, col(col_name).cast(FloatType()))
        test_df = test_df.withColumn(col_name, col(col_name).cast(FloatType()))

    # Melt train df into long format
    train_long = train_df.select("SubjectID", "EpochID", "label", *feature_cols) \
        .selectExpr("SubjectID", "EpochID", "label", "stack({0}, {1}) as (pivot, value)".format(
            len(feature_cols),
            ', '.join([f"'{c}', `{c}`" for c in feature_cols])
        ))

    # Compute normalization stats
    stats = train_long.groupBy("pivot").agg(
        _mean("value").alias("mean_val"),
        _stddev("value").alias("std_val")
    )

    # Normalize and pivot back (can be slow for large data)
    def apply_normalization(df):
        long_df = df.select("SubjectID", "EpochID", "label", *feature_cols) \
            .selectExpr("SubjectID", "EpochID", "label", "stack({0}, {1}) as (pivot, value)".format(
                len(feature_cols),
                ', '.join([f"'{c}', `{c}`" for c in feature_cols])
            ))
        long_df = long_df.join(broadcast(stats), on="pivot")
        long_df = long_df.withColumn(
            "norm_value",
            (col("value") - col("mean_val")) / when((col("std_val").isNotNull()) & (col("std_val") != 0), col("std_val")).otherwise(1.0)
        )
        return long_df.groupBy("SubjectID", "EpochID", "label").pivot("pivot").agg(first("norm_value")).fillna(0.0)

    return apply_normalization(train_df), apply_normalization(test_df)



    # normalizes per subject for columns
def normalize_by_column_per_subject_wide(df, feature_cols):
    from pyspark.sql.functions import mean as _mean, stddev as _stddev, col, when
    stats = df.groupBy("SubjectID").agg(
        *[
            _mean(c).alias(f"{c}_mean") for c in feature_cols
        ] + [
            _stddev(c).alias(f"{c}_std") for c in feature_cols
        ]
    )

    # Join stats back to original
    df = df.join(stats, on="SubjectID", how="left")

    # Normalize each column with its respective mean and std per subject
    for col_name in feature_cols:
        mean_col = f"{col_name}_mean"
        std_col = f"{col_name}_std"
        df = df.withColumn(
            col_name,
            (col(col_name) - col(mean_col)) / when((col(std_col).isNotNull()) & (col(std_col) != 0), col(std_col)).otherwise(1.0)
        )

    # Drop the extra mean/std columns used just for normalization
    return df.drop(*[f"{c}_mean" for c in feature_cols], *[f"{c}_std" for c in feature_cols])


def min_max_by_column_per_subject_wide(df, feature_cols):
    from pyspark.sql.functions import min as _min, max as _max, col, when
    # Compute min and max per SubjectID and feature
    stats = df.groupBy("SubjectID").agg(
        *[
            _min(c).alias(f"{c}_min") for c in feature_cols
        ] + [
            _max(c).alias(f"{c}_max") for c in feature_cols
        ]
    )

    # Join the stats back to the original dataframe
    df = df.join(stats, on="SubjectID", how="left")

    # Apply min-max normalization: (x - min) / (max - min)
    for col_name in feature_cols:
        min_col = f"{col_name}_min"
        max_col = f"{col_name}_max"
        df = df.withColumn(
            col_name,
            (col(col_name) - col(min_col)) /
            when((col(max_col) != col(min_col)) & col(max_col).isNotNull(), col(max_col) - col(min_col)).otherwise(1.0)
        )

    # Drop the temporary min/max columns
    return df.drop(*[f"{c}_min" for c in feature_cols], *[f"{c}_max" for c in feature_cols])



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

def apply_pca_model(target_df, pca_cols, pca_model, k):
 
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql.functions import col
    from pyspark.sql.types import IntegerType


   # Step 1: Assemble input features into a single vector
    assembler = VectorAssembler(inputCols=pca_cols, outputCol="features")
    assembled_df = assembler.transform(target_df)

    # Step 2: Apply PCA model
    transformed = pca_model.transform(assembled_df)

    # Step 3: Select PCA features along with SubjectID, EpochID, label
    final_df = transformed.select(
        col("SubjectID"),
        col("EpochID"),
        col("label").cast(IntegerType()).alias("label"),
        col("pca_features").alias("features")
    )

    return final_df


#the function below doens't include subject and epochid in the final_df

# def apply_pca_model(target_df, pca_cols, pca_model, k):
#     # Step 1: Assemble original features (before PCA) into a vector
#     assembler = VectorAssembler(inputCols=pca_cols, outputCol="features")
#     assembled_df = assembler.transform(target_df)
#
#     # Step 2: Apply the PCA model
#     transformed = pca_model.transform(assembled_df)
#
#     # Step 3: Extract just the PCA output vector and label
#     # Ensure label is IntegerType
#     final_df = transformed.select(
#         col("pca_features").alias("features"),
#         col("label").cast(IntegerType()).alias("label")
#     )
#
#     return final_df


# ********************************* !! DEPRECATED !! ***********************************************

# this z-scores all the data between 2 datasets - tested works 
def normalize_power(train_spark_df, test_spark_df):
    from pyspark.sql.functions import mean as _mean, stddev as _stddev, broadcast, when
    from pyspark.sql.types import FloatType

    train_spark_df = train_spark_df.withColumn("Power", col("Power").cast(FloatType())) # cast as a 4 byte number 
    test_spark_df = test_spark_df.withColumn("Power", col("Power").cast(FloatType()))

    # Notice how our stats are only from the training group
    stats = train_spark_df.groupBy("Electrode", "WaveBand").agg(
        _mean("Power").alias("mean_power"),
        _stddev("Power").alias("std_power")
    )

    #2 more rows created mean and std power for each electrod and waveband
    train_spark_df = train_spark_df.join(broadcast(stats), on=["Electrode", "WaveBand"]) 
    test_spark_df = test_spark_df.join(broadcast(stats), on=["Electrode", "WaveBand"])

    # the .otherwise 1.0 makes it so that if the std power is 0 or null , then make it 1 , 
    # if std is 0 or null it means no varience in that variable (currnt_val == avg) and  so the result(s) in that column will be 0 as (currnet_val - mean) / 1 = 0
    train_spark_df = train_spark_df.withColumn("Power", (col("Power") - col("mean_power")) / when((col("std_power").isNotNull()) & (col("std_power") != 0), col("std_power")).otherwise(1.0))
    test_spark_df = test_spark_df.withColumn("Power", (col("Power") - col("mean_power")) / when((col("std_power").isNotNull()) & (col("std_power") != 0), col("std_power")).otherwise(1.0))

    # drop the mean and the std_power 
    train_spark_df = train_spark_df.drop("mean_power", "std_power")
    test_spark_df = test_spark_df.drop("mean_power", "std_power")
    return train_spark_df, test_spark_df



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

