import subprocess


def test_bloom_filter():
    result = subprocess.run(
        [
            "python",
            "-m",
            "text_dedup.bloom_filter",
            "--path",
            "allenai/c4",
            "--name",
            "xh",
            "--split",
            "train",
            "--cache_dir",
            ".cache",
            "--output",
            ".temp-output",
            "--column",
            "text",
            "--batch_size",
            "10000",
        ],
        capture_output=True,
        text=True,
    )

    # check the output
    assert (
        "69048" in result.stdout and "69048" in result.stdout
    ), f"Expected before and after are not present in the output: {result.stdout}"

    # remove the output and input
    subprocess.run(["rm", "-rf", ".cache"])
    subprocess.run(["rm", "-rf", ".temp-output"])
