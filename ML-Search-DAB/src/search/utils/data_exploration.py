from pyspark.sql import functions as F
from pyspark.sql.window import Window


def drop_docs_analysis(df):
    window_spec = Window.partitionBy(F.col('queryId')).orderBy(F.col('resPos'))
    df = (df.withColumn('prev_score', F.lag(F.col('finalScore'), 1).over(window_spec))
                .withColumn('max_score', F.max(F.col('finalScore')).over(Window.partitionBy(F.col('queryId'))))
                .withColumn('min_score', F.min(F.col('finalScore')).over(Window.partitionBy(F.col('queryId'))))
                .withColumn('drop', F.when(
        (F.col('prev_score').isNull()) | (F.col('min_score') == F.col('max_score')), 0
    ).otherwise(
        (F.col('prev_score') - F.col('finalScore')) / (F.col('max_score') - F.col('min_score'))
    ))
    .withColumn('first_label_0', F.when(F.col('drop') > 0.5, F.col('finalScore')).otherwise(F.lit(0)))
    .withColumn('max_first_label_0', F.max(F.col('first_label_0')).over(Window.partitionBy(F.col('queryId'))))
    .withColumn('keep_label', F.when(F.col('finalScore') > F.col('max_first_label_0'), F.lit(1)).otherwise(F.lit(0))))
    columns_to_drop = ['prev_score', 'max_score', 'min_score', 'drop', 'first_label_0', 'max_first_label_0']
    df = df.select([col for col in df.columns if col not in columns_to_drop])
    return df


def compare_docs_with_clicks(df) -> None:
    # Calculate the total sums
    total_sums = df.agg(
        F.sum('startDocs').alias('totalstartDocs'),
        F.sum('keepDocs').alias('totalkeepDocs'),
        F.sum('startClicks').alias('totalStartClicks'),
        F.sum('keepClicks').alias('totalKeepClicks'),
    ).collect()[0]

    # Calculate the percentage
    total_start_doc_count = total_sums['totalstartDocs']
    total_final_doc_count = total_sums['totalkeepDocs']
    total_start_clk_count = total_sums['totalStartClicks']
    total_final_clk_count = total_sums['totalKeepClicks']

    percentage = (1 - total_final_doc_count / total_start_doc_count) * 100
    print(f"Total Dropped Doc%: {percentage:.2f}%")

    if total_start_clk_count > 0:
        percentage = (1 - total_final_clk_count / total_start_clk_count) * 100
        print(f"Total Dropped Clicks%: {percentage:.2f}%")
    else:
        print("No clicks were dropped.")

    print("------------------------------------------------------------")
    percentage = (total_start_clk_count / total_start_doc_count) * 100
    print(f"CTR before: {percentage:.2f}%")
    percentage = (total_final_clk_count / total_final_doc_count) * 100
    print(f"CTR after: {percentage:.2f}%")
    return None