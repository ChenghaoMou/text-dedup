import subprocess  # nosec


def test_minhash():
    result = subprocess.run(
        [
            "python",
            "-m",
            "text_dedup.ann_unisim",
            "--path",
            "truthful_qa",
            "--name",
            "generation",
            "--split",
            "validation",
            "--cache_dir",
            ".cache",
            "--output",
            ".temp-output",
            "--column",
            "question",
            "--batch_size",
            "24",
        ],
        capture_output=True,
        text=True,
    )  # nosec

    # check the output
    assert "817" in result.stdout and "788" in result.stdout, (
        f"Expected before and after are not present in the output: {result.stdout}"
    )

    # remove the output and input
    # subprocess.run(["rm", "-rf", ".cache"])  # nosec
    subprocess.run(["rm", "-rf", ".temp-output"])  # nosec
