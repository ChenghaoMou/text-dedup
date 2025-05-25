import inspect
import os
import random
from pathlib import Path

import click
import numpy as np
from onnxruntime import InferenceSession
from tqdm import tqdm
from unisim import TextSim
from unisim.embedder import Embedder

from text_dedup import logger
from text_dedup.utils import CLUSTER_COLUMN
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import UniSimArgs
from text_dedup.utils import load_hf_dataset
from text_dedup.utils import random_samples

EMBEDDING_COLUMN = "__embeddings__"


class WrapInferenceSession:
    def __init__(self, *args, **kwargs):
        self.sess = InferenceSession(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {"args": self.args, "kwargs": self.kwargs}

    def __setstate__(self, values):
        self.args = values["args"]
        self.kwargs = values["kwargs"]
        self.sess = InferenceSession(*self.args, **self.kwargs)


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

    # A workaround to enable multiprocessing for the inference session
    fpath = Path(inspect.getfile(Embedder))
    mpath = fpath.parent / "models" / unisim_args.model_id
    providers = ["CPUExecutionProvider"] if not unisim_args.use_accelerator else ["CUDAExecutionProvider"]
    sess = WrapInferenceSession(str(mpath.with_suffix(".onnx")), providers=providers)
    text_sim.embedder.model["sess"] = sess

    with timer("Total"):
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)

        with timer("Embedding"):
            ds = ds.map(
                lambda batch: {
                    EMBEDDING_COLUMN: text_sim.embedder.embed(batch[meta_args.column]),
                },
                num_proc=io_args.num_proc,
                batched=True,
                batch_size=meta_args.batch_size,
                load_from_cache_file=True,
            )

        LEN_DATASET = len(ds)
        NUM_SHARDS = np.ceil(LEN_DATASET / meta_args.batch_size).astype(int)
        text_sim._lazy_init()

        with timer("Indexing"):
            for batch_idx in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Indexing embeddings...",
            ):
                shard = ds.shard(
                    num_shards=NUM_SHARDS, index=batch_idx, contiguous=True, writer_batch_size=meta_args.batch_size
                )
                batch_indices = shard[INDEX_COLUMN]
                batch_embedds = shard[EMBEDDING_COLUMN]
                text_sim.indexer.add(batch_embedds, batch_indices)
                if unisim_args.store_data:
                    text_sim.indexed_data.extend(shard[meta_args.column])

        with timer("Querying"):
            results = []

            for batch_idx in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Querying embeddings...",
            ):
                shard = ds.shard(
                    num_shards=NUM_SHARDS, index=batch_idx, contiguous=True, writer_batch_size=meta_args.batch_size
                )

                remain_embedds = shard[EMBEDDING_COLUMN]
                remain_indices = shard[INDEX_COLUMN]
                shard_results = [[] for _ in remain_indices]
                k = 20
                while remain_embedds and remain_indices:
                    res = text_sim.indexer.search(
                        queries=[_ for _ in remain_indices],
                        query_embeddings=np.asarray(remain_embedds),
                        similarity_threshold=unisim_args.similarity_threshold,
                        k=k,
                        drop_closest_match=True,
                        return_data=unisim_args.store_data,
                        return_embeddings=unisim_args.return_embeddings,
                        data=text_sim.indexed_data,
                    )
                    res = [
                        [m for m in r.matches if m.similarity >= unisim_args.similarity_threshold] for r in res.results
                    ]
                    unfinished = []
                    for i, r in enumerate(res):
                        if r and len(r) == k:
                            unfinished.append(i)
                        else:
                            shard_results[i].extend(r)

                    remain_indices = [remain_indices[i] for i in unfinished]
                    remain_embedds = [remain_embedds[i] for i in unfinished]

                    k *= 2

                results.extend(zip(shard[INDEX_COLUMN], shard_results))

        with timer("Clustering"):
            for idx, matches in tqdm(results):
                for match in matches:
                    uf.union(idx, match.idx)

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=io_args.num_proc,  # type: ignore
                new_fingerprint=str(random.getrandbits(128)),  # type: ignore
                desc="Finding clusters...",  # type: ignore
            )
            final_data = ds.filter(
                function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
                with_indices=False,
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
                uf.dump(os.path.join(io_args.output, "uf.pkl"), id2id=id2id)

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
