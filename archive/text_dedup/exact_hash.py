#!/usr/bin/env python
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import multiprocessing as mp
from typing import Callable

import click
import numpy as np
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import ExactHashArgs
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils.hashfunc import md5_hexdigest
from text_dedup.utils.hashfunc import sha256_hexdigest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.load import load_hf_dataset
from text_dedup.utils.memory import DisableReferenceCount
from text_dedup.utils.timer import Timer

mp.set_start_method("fork", force=True)


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@ExactHashArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    exact_hash_args: ExactHashArgs,
):
    timer = Timer()

    # we use the hex digests for md5 and sha256 for legacy compatibility reasons
    # we use the raw xxh3_128 byte digests for speed
    hash_func: Callable = {
        "md5": md5_hexdigest,  # type: ignore
        "sha256": sha256_hexdigest,  # type: ignore
        "xxh3": xxh3_128_digest,  # type: ignore
    }[exact_hash_args.hash_func]
    hashes = set()
    flags = []

    with timer("Total"):
        with timer("Loading"):
            ds, _ = load_hf_dataset(io_args=io_args, meta_args=meta_args)

        LEN_DATASET: int = len(ds)
        NUM_SHARDS = int(np.ceil(LEN_DATASET / meta_args.batch_size))

        with timer("Processing"):
            # currently processing is done on a single thread.
            # still, due to the nature of the calculations it is O(len(ds))
            # to make multithreaded, would have to handle shared data structs etc.
            # most approaches are not low hanging fruit.
            for idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                ds_shard = ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True)
                for example in tqdm(ds_shard[meta_args.column], leave=False):
                    # moving this byte conversion outside the loop saw no improvement <1 GiB datasets
                    # might not be worth the added overhead
                    h = hash_func(example.encode("utf-8"))
                    if h in hashes:
                        flags.append(True)
                    else:
                        flags.append(False)
                        hashes.add(h)

        with timer("Filtering"), DisableReferenceCount():
            # batch size here would be a trade off between memory and speed
            # default is 1000
            ds = ds.filter(
                lambda _, idx: not flags[idx],
                with_indices=True,
                num_proc=io_args.num_proc,
                writer_batch_size=meta_args.batch_size,
            )

        with timer("Saving"):
            ds.save_to_disk(io_args.output)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
