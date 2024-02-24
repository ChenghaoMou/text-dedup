import subprocess


def test_suffix_array():
    result = subprocess.run(
        [
            "python",
            "-m",
            "text_dedup.suffix_array",
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
            "--google_repo_path",
            "./deduplicate-text-datasets",
        ],
        capture_output=True,
        text=True,
    )

    # check the output
    assert (
        "180332342 bytes (88803)" in result.stdout
        and "51305898 bytes (29254)" in result.stdout
    ), f"Expected before and after are not present in the output: {result.stdout}"
