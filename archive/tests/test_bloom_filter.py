import io
import subprocess  # nosec
from contextlib import redirect_stdout

import click

from text_dedup.bloom_filter import main as bf_main
from text_dedup.utils import BloomFilterArgs
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs


def test_bloom_filter():
    with redirect_stdout(io.StringIO()) as f:
        ctx = click.Context(bf_main)
        ctx.invoke(
            bf_main,
            io_args=IOArgs(
                path="allenai/c4",
                name="xh",
                split="train",
                cache_dir=".cache",
                output=".temp-output",
            ),
            meta_args=MetaArgs(column="text", batch_size=10000),
            bloom_filter_args=BloomFilterArgs(),
        )
    s = f.getvalue()
    # check the output
    print(f"Output:\n{s}")
    assert "69048" in s and "69048" in s, f"Expected before and after are not present in the output: {s}"

    # remove the output and input
    # subprocess.run(["rm", "-rf", ".cache"])  # nosec
    subprocess.run(["rm", "-rf", ".temp-output"])  # nosec
