from pyspark.sql.functions import col, concat_ws, lit
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.functions import vector_to_array


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
