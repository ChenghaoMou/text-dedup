import os
import subprocess  # nosec

import datasets


def test_minhash():
    ds = datasets.load_dataset("allenai/c4", "xh", split="train")
    # convert to pandas
    df = ds.to_pandas()

    os.makedirs("./temp-data", exist_ok=True)
    os.makedirs("./temp-output", exist_ok=True)

    df.to_parquet("./temp-data/temp-data.parquet")

    args = """
    spark-submit --executor-memory 86g
        --driver-memory 8g
        --executor-cores 2
        --num-executors 2
        --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12
        --conf spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
        --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
        text_dedup/minhash_spark.py
        --input ./temp-data
        --output ./temp-output
        --column text
        --threshold 0.7
    """.split("\n")
    result = subprocess.run(
        [part.strip() for line in args for part in line.strip().split(" ") if part.strip()],
        capture_output=True,
        text=True,
    )  # nosec

    # check the output
    print(f"Output:\n{result.stdout}")
    assert "68436" in result.stdout and "66529" in result.stdout, (
        f"Expected before and after are not present in the output: {result.stdout}"
    )

    # remove the output and input
    # subprocess.run(["rm", "-rf", ".cache"])  # nosec
    subprocess.run(["rm", "-rf", ".temp-output"])  # nosec
