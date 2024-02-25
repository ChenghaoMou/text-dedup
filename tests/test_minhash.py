import subprocess


def test_minhash():
    result = subprocess.run(
        [
            "python",
            "-m",
            "text_dedup.minhash",
            "--path",
            "oscar-corpus/OSCAR-2201",
            "--name",
            "gl",
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
        "88803" in result.stdout and "44092" in result.stdout
    ), f"Expected before and after are not present in the output: {result.stdout}"

    # remove the output and input
    subprocess.run(["rm", "-rf", ".cache"])
    subprocess.run(["rm", "-rf", ".temp-output"])
