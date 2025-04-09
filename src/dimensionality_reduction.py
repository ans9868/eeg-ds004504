from pyspark.sql.functions import col, concat_ws, lit
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import IntegerType


# TODO: make config for variance target 


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



