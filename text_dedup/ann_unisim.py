import os
import pickle  # nosec
import random

import click
import numpy as np
from tqdm import tqdm
from unisim import TextSim

from text_dedup import logger
from text_dedup.utils.args import IOArgs
from text_dedup.utils.args import MetaArgs
from text_dedup.utils.args import UniSimArgs
from text_dedup.utils.inspect import random_samples
from text_dedup.utils.load import load_hf_dataset
from text_dedup.utils.memory import DisableReferenceCount
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find import UnionFind


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@UniSimArgs.option_group
def main(io_args: IOArgs, meta_args: MetaArgs, unisim_args: UniSimArgs):
    timer = Timer()
    uf = UnionFind()

    text_sim = TextSim(
        store_data=unisim_args.store_data,
        index_type=unisim_args.index_type,
        return_embeddings=unisim_args.return_embeddings,
        model_id=unisim_args.model_id,
        index_params=unisim_args.index_params,
        batch_size=unisim_args.batch_size,
        use_accelerator=unisim_args.use_accelerator,
    )

    with timer("Total"):
        with timer("Load Dataset"):
            ds = load_hf_dataset(io_args)
            ds = ds.map(lambda x, i: {"__idx__": i}, with_indices=True, num_proc=io_args.num_proc)

        LEN_DATASET = len(ds)
        NUM_SHARDS = np.ceil(LEN_DATASET / meta_args.batch_size).astype(int)

        with timer("Index Dataset"):
            for batch_idx in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Iterating Embeddings...",
            ):
                # Iterate over each batch dataset from the total hash embedded dataset
                shard = ds.shard(
                    num_shards=NUM_SHARDS, index=batch_idx, contiguous=True, writer_batch_size=meta_args.batch_size
                )
                text_sim.add(shard[meta_args.column])

        with timer("Clustering"):
            for idx in tqdm(range(len(ds))):
                # ! Since there is no easy way to exhaustively search all candidates with a given similarity_threshold,
                # ! we use a simple heuristic to find the all the cnadidates with a back-off strategy.
                # TODO: this could be a bottlebeck for large datasets.
                results = []
                k = 20
                while True:
                    res = text_sim.search(
                        [ds[idx][meta_args.column]],
                        similarity_threshold=unisim_args.similarity_threshold,
                        drop_closest_match=True,
                        k=k,
                    )
                    curr_res = [m for m in res.results[0].matches if m.similarity >= unisim_args.similarity_threshold]
                    if curr_res and len(curr_res) == k:
                        k *= 2
                        continue
                    results = curr_res
                    break
                for match in results:
                    uf.union(match.idx, idx)

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda _, idx: {"__cluster__": uf.find(idx)},
                with_indices=True,
                num_proc=io_args.num_proc,  # type: ignore
                new_fingerprint=str(random.getrandbits(128)),  # type: ignore
                desc="Finding clusters...",  # type: ignore
            )
            final_data = ds.filter(
                function=lambda record, idx: record["__cluster__"] == idx,
                with_indices=True,
                num_proc=io_args.num_proc,
                desc="Filtering clusters...",
            )
            if io_args.debug:
                # ! Expensive operation, but useful for debugging.
                random_samples(ds, cluster_column="__cluster__", text_column=meta_args.column)

        with timer("Saving"):
            final_data = final_data.remove_columns(["__cluster__"])
            final_data.save_to_disk(io_args.output)
            if io_args.debug:
                with open(os.path.join(io_args.output, "uf.pkl"), "wb") as f:
                    pickle.dump(uf, f, protocol=pickle.HIGHEST_PROTOCOL)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")


if __name__ == "__main__":
    main()
