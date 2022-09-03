import hashlib
import json
import logging
import os
import sys
import textwrap
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import hydra
import numpy as np
import rich
from datasets import (get_dataset_config_names, get_dataset_split_names,
                      load_dataset)
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.embedders.simhash import SimHashEmbedder
from text_dedup.embedders.suffix import SuffixArrayEmbedder
from text_dedup.postprocess.clustering import (lsh_clustering,
                                               simhash_clustering)
from text_dedup.utils.hf_datasets.helpers import (dataset_get,
                                                  dataset_get_all_str_columns,
                                                  dataset_map, extract_text)


def __clear_screen():  # pragma: no cover
    sys.stderr.flush()
    sys.stdout.flush()


def __disable_logging(library: str):
    logging.getLogger(library).setLevel(logging.ERROR)


console = Console()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
TOKEN = os.environ.get("HF_ACCESS_TOKEN", True)
SPLITS: List[str] = ["train", "validation", "test"]
logger: logging.Logger = logging.getLogger(
    "text_dedup",
)


def __get_slice_text(text: str, byte_slice: slice) -> str:  # pragma: no cover
    return text.encode("utf-8")[byte_slice].decode("utf-8", errors="ignore")


def __compute_md5(obj: Any) -> str:  # pragma: no cover
    if isinstance(obj, DictConfig):
        return hashlib.md5(OmegaConf.to_yaml(obj).encode("utf-8")).hexdigest()
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def __get_default_embedder(conf: DictConfig) -> Tuple[
    Union[MinHashEmbedder, SimHashEmbedder],
    Callable
]:
    embedder: Union[SimHashEmbedder, MinHashEmbedder] = SimHashEmbedder()
    if conf.embedder.name == "MinHashEmbedder":
        embedder = MinHashEmbedder(num_perm=conf.embedder.num_perm)

    embed_function = embedder.embed_function(
        n_gram=conf.tokenization.ngram_size,
        level=conf.tokenization.level,
        use_str=conf.embedder.name == "SimHashEmbedder",
    )

    return embedder, embed_function


@hydra.main(config_name="config", config_path="configs", version_base="1.2")
def main(conf: DictConfig):  # pragma: no cover
    conf = conf.method

    conf.cache_dir = conf.cache_dir.rstrip("/")

    if not os.path.exists(conf.cache_dir):
        os.makedirs(conf.cache_dir, exist_ok=True)
    if not os.path.exists(Path(conf.output).parent):
        os.makedirs(Path(conf.output).parent, exist_ok=True)

    num_proc: int = min((conf.num_proc or 1), os.cpu_count())  # type: ignore
    use_streaming: bool = num_proc == 1

    if not conf.configs:
        conf.configs = get_dataset_config_names(conf.dataset, use_auth_token=TOKEN)
        logger.warning(
            f"No configs specified, using all available configs {conf.configs}")
    else:
        logger.info(f"Using configs {conf.configs}")

    if conf.embedder.name not in {
        "SimHashEmbedder",
        "MinHashEmbedder",
        "SuffixArrayEmbedder",
    }:
        raise ValueError(f"Unknown embedder {conf.embedder.name}")

    for config in conf.configs:
        splits = get_dataset_split_names(conf.dataset, config, use_auth_token=TOKEN)
        splits = [s for s in SPLITS if s in splits]

        split_results: Dict[str, Any] = {}

        cm_table = Table(title=f"{config} results")
        cm_table.add_column("Split", justify="center", style="cyan", no_wrap=True)
        for split in splits:
            cm_table.add_column(split, justify="center", style="magenta", no_wrap=True)

        conf_columns: List[str] = list(conf.columns) or []

        for split in splits:

            split_data = load_dataset(
                conf.dataset,
                config,
                split=split,
                use_auth_token=TOKEN,
                cache_dir=conf.cache_dir if not use_streaming else None,
                streaming=use_streaming,
            )

            if not conf_columns:
                conf_columns = dataset_get_all_str_columns(split_data)
            if not conf_columns:
                raise ValueError(f"No columns specified in {split}")

            __clear_screen()
            logger.info(f"Using columns in {split}: {conf_columns}")

            if conf.embedder.name in {"SimHashEmbedder", "MinHashEmbedder"}:
                embedder, embed_function = __get_default_embedder(conf)

                def fingerprinting(x):
                    y = extract_text(x, conf_columns)
                    res = {
                        "__signature__": embed_function(y["__text__"]),
                        "__size__": y["__size__"],
                    }
                    if not use_streaming:
                        res["__text__"] = y["__text__"]
                    return res

                split_data = dataset_map(
                    split_data,
                    function=lambda x: fingerprinting(x),
                    num_proc=num_proc,
                    remove_columns=split_data.column_names if not use_streaming else None,
                    desc=f"Fingerprinting {split}",
                )
            else:
                split_data = dataset_map(
                    split_data,
                    function=lambda x: extract_text(x, conf_columns),
                    num_proc=num_proc,
                    remove_columns=split_data.column_names if not use_streaming else None,
                    desc=f"Extracting text from {split}",
                )

            split_results[split] = split_data

        __clear_screen()
        records: List[Dict[str, Any]] = []
        table_rows = {x: {y: "" for y in SPLITS} for x in SPLITS}

        if conf.embedder.name in {"SimHashEmbedder", "MinHashEmbedder"}:
            # All pair combinations
            for x, y in product(splits, repeat=2):
                if SPLITS.index(x) > SPLITS.index(y):
                    continue

                __clear_screen()
                logger.info(f"Looking for {y}'s duplicates in {x}")

                base_data = split_results[x]
                query_data = split_results[y]

                clustering_config_md5 = __compute_md5(
                    {
                        "x": x,
                        "y": y,
                        "config": config,
                        "dataset": conf.dataset,
                        "columns": conf_columns,
                        "ngram_size": conf.tokenization.ngram_size,
                        "level": conf.tokenization.level,
                        "embedder": repr(embedder),
                    }
                )
                # Datasketch takes bytes as the basename
                clustering_config_md5 = clustering_config_md5.encode("utf-8") if \
                    conf.embedder.name == "MinHashEmbedder" else clustering_config_md5  # type: ignore
                storage_config = {
                    "type": conf.storage_config.type,
                    "redis": {
                        "host": conf.storage_config.redis.host,
                        "port": conf.storage_config.redis.port,
                    },
                    "basename": clustering_config_md5,
                } if conf.storage_config and conf.storage_config.type == "redis" else None

                if conf.embedder.name == "SimHashEmbedder":
                    clusters: List[List[int]] = simhash_clustering(
                        signatures=dataset_get(base_data, "__signature__", int),
                        hamming_distance=conf.embedder.hamming_distance,
                        query_signatures=dataset_get(query_data, "__signature__", int),
                        num_blocks=conf.embedder.num_blocks,
                        storage_config=storage_config,
                        verbose=conf.verbose,
                    )
                else:
                    clusters = lsh_clustering(
                        signatures=dataset_get(base_data, "__signature__", np.asarray),
                        threshold=conf.embedder.threshold,
                        query_signatures=dataset_get(
                            query_data, "__signature__", np.asarray),
                        storage_config=storage_config,
                        verbose=conf.verbose,
                    )

                # Collect stats
                duplicated_count = 0
                total_count = 0
                duplicated_size = 0
                total_size = 0
                examples = 5

                for i, (cluster, size_) in enumerate(tqdm(zip(
                        clusters, dataset_get(query_data, "__size__")
                ), total=len(clusters), desc="Post-processing...")):
                    total_count += 1
                    total_size += size_
                    if len(cluster) <= 1:  # No duplicates, a.k.a. the only duplicate is itself
                        continue
                    duplicated_count += 1
                    duplicated_size += size_
                    cluster = [j for j in cluster if (x, j) != (y, i)]
                    records.append(
                        {
                            "query_index": i,
                            "query_split": y,
                            "references": cluster,
                            "reference_split": x,
                        }
                    )
                    # Showcase some examples
                    if examples > 0 and not use_streaming:
                        table = Table(title="Examples", show_lines=True)
                        table.add_column("Query Split", justify="left",
                                         style="cyan", no_wrap=False)
                        table.add_column("Query Index", justify="left",
                                         style="cyan", no_wrap=False)
                        table.add_column("Query Instance", justify="left",
                                         style="cyan", no_wrap=False)
                        table.add_column("Duplicate Split", justify="left",
                                         style="cyan", no_wrap=False)
                        table.add_column("Duplicate Index", justify="left",
                                         style="cyan", no_wrap=False)
                        table.add_column("Duplicate", justify="left", style="magenta")
                        for ref_id, reference in zip(cluster[:10], base_data.select(cluster)["__text__"]):
                            table.add_row(
                                y,
                                str(i),
                                textwrap.shorten(query_data.select(
                                    [i])["__text__"][0], width=512),
                                x,
                                str(ref_id),
                                textwrap.shorten(reference, width=512),
                            )
                        try:
                            print(table)
                        except rich.errors.MarkupError:
                            continue
                        examples -= 1

                duplicated_count_ratio: float = duplicated_count / total_count * 100
                duplicated_byte_ratio: float = duplicated_size / total_size * 100

                final_count = f"N {duplicated_count_ratio:.2f}% ({duplicated_count})"
                final_byte = f"B {duplicated_byte_ratio:.2f}% ({duplicated_size})"
                table_rows[x][y] = f"{final_count} | {final_byte}"

        elif conf.embedder.name == "SuffixArrayEmbedder":
            # All pair combinations
            embedder = SuffixArrayEmbedder(k=conf.embedder.k)  # type: ignore
            for x, y in product(splits, repeat=2):
                if SPLITS.index(x) > SPLITS.index(y):
                    continue

                __clear_screen()
                logger.info(f"Looking for {y}'s duplicates in {x}")

                base_data = split_results[x]
                query_data = split_results[y]

                slices_x, slices_y = embedder.google_embed(  # type: ignore
                    base_data["__text__"],
                    query_data["__text__"] if x != y else None,
                    merge=True,
                    merge_strategy="longest",
                    skip_existing=conf.embedder.skip_existing,
                    cache_dir=conf.embedder.cache_dir,
                    temp_file_prefix=conf.embedder.temp_file_prefix,
                )

                # Collect stats
                duplicated_count = 0
                total_count = 0
                duplicated_size = 0
                total_size = 0

                def process_slices(slices, sizes, query_split, reference_split):
                    nonlocal total_size, total_count, duplicated_size, duplicated_count, records

                    for idx, (segments, size) in enumerate(zip(slices, sizes)):
                        total_size += size
                        total_count += 1
                        duplicated_count += 1 if segments else 0
                        duplicated_size += sum(s.stop
                                               - s.start for s in segments) if segments else 0

                        records.append(
                            {
                                "query_index": idx,
                                "query_split": query_split,
                                "reference_split": reference_split,
                                "byte_slices": [[s.start, s.stop] for s in segments],
                            }
                        )

                process_slices(slices_x, base_data["__size__"], x, y)

                if x != y:
                    process_slices(slices_y, query_data["__size__"], y, x)

                duplicated_count_ratio = duplicated_count / total_count * 100
                duplicated_byte_ratio = duplicated_size / total_size * 100

                final_count = f"N {duplicated_count_ratio:.2f}% ({duplicated_count})"
                final_byte = f"B {duplicated_byte_ratio:.2f}% ({duplicated_size})"

                table_rows[x][y] = f"{final_count} | {final_byte}"

        with open(f"{conf.output}", "w") as f:
            json.dump(records, f)

        for x in splits:
            cm_table.add_row(x, *[table_rows[x][y] or "" for y in splits])

        __clear_screen()
        console.print(cm_table)


if __name__ == "__main__":
    import time
    import tracemalloc

    try:
        from humanize import naturalsize
    except ImportError:
        def naturalsize(x):
            return x

    tracemalloc.start()

    start_time = time.time()

    main()

    _, usage = tracemalloc.get_traced_memory()

    logger.info(f"Done in {time.time() - start_time:.2f} seconds")
    logger.info(f"Using {naturalsize(usage)} memory")

    tracemalloc.stop()
