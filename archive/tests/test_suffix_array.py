import subprocess  # nosec


# @pytest.mark.skip(reason="This test is too slow")
def test_suffix_array():
    result = subprocess.run(
        [
            "python",
            "-m",
            "text_dedup.suffix_array",
            "--path",
            "allenai/c4",
            "--name",
            "yo",
            "--split",
            "train",
            "--cache_dir",
            ".cache",
            "--output",
            ".temp-output",
            "--column",
            "text",
            "--num_proc",
            "7",
            "--google_repo_path",
            "./deduplicate-text-datasets",
        ],
        capture_output=True,
        text=True,
    )  # nosec

    # check the output
    print(f"Output:\n{result.stdout}")
    assert "155279898 bytes (46214)" in result.stdout and "140874800 bytes (46149)" in result.stdout, (
        f"Expected before and after are not present in the output: {result.stdout}"
    )

    # remove the output and input
    # subprocess.run(["rm", "-rf", ".cache"])  # nosec
    subprocess.run(["rm", "-rf", ".temp-output"])  # nosec
