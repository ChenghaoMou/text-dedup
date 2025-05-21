import subprocess  # nosec


def test_exact_hash():
    result = subprocess.run(
        [
            "python",
            "-m",
            "text_dedup.exact_hash",
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
    )  # nosec

    # check the output
    print(f"Output:\n{result.stdout}")
    assert "69048" in result.stdout and "69048" in result.stdout, (
        f"Expected before and after are not present in the output: {result.stdout}"
    )

    # remove the output and input
    # subprocess.run(["rm", "-rf", ".cache"])  # nosec
    subprocess.run(["rm", "-rf", ".temp-output"])  # nosec
