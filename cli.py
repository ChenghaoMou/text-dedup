import hashlib
import json
import logging
import os
import sys
import textwrap
import time
from itertools import product
from typing import Any, Dict, List, Union

import hydra
from datasets import Value, get_dataset_config_names, get_dataset_split_names, load_dataset
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.table import Table
from tqdm import tqdm

from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.embedders.simhash import SimHashEmbedder
from text_dedup.embedders.suffix import SuffixArrayEmbedder
from text_dedup.utils.clustering import lsh_clustering, simhash_clustering

TOKEN = os.environ.get("HF_ACCESS_TOKEN", True)


def disable_logging(library: str):
    logging.getLogger(library).setLevel(logging.ERROR)


logger: logging.Logger = logging.getLogger(
    "text_dedup",
)
SPLITS: List[str] = ["train", "validation", "test"]


def clear_screen():  # pragma: no cover
    sys.stderr.flush()
    sys.stdout.flush()


def get_byte_size(x: str) -> int:  # pragma: no cover
    return sys.getsizeof(x)


def get_slice_text(text: str, byte_slice: slice) -> str:  # pragma: no cover
    return text.encode("utf-8")[byte_slice].decode("utf-8", errors="ignore")


def compute_md5(obj: Any) -> str:  # pragma: no cover
    if isinstance(obj, DictConfig):
        return hashlib.md5(OmegaConf.to_yaml(obj).encode("utf-8")).hexdigest()
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


@hydra.main(config_name="config", config_path="configs", version_base="1.2")
def main(conf: DictConfig):  # pragma: no cover
    start_time = time.time()
    conf = conf.method
    conf.cache_dir = conf.cache_dir.rstrip("/")
    storage_prefix: str = f"{conf.cache_dir}/{conf.dataset.replace('/', '_')}"
    num_proc: int = conf.num_proc or os.cpu_count() or 1

    if not conf.configs:
        conf.configs = get_dataset_config_names(conf.dataset, use_auth_token=TOKEN)
        logger.warning(f"No configs specified, using all available configs {conf.configs}")
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

        def extract_text(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
            text = " ".join(str(row[f]) for f in columns)
            return {"__text__": text, "__size__": get_byte_size(text)}

        split_results: Dict[str, Any] = {}
        conf_columns: List[str] = list(conf.columns) or []

        for split in splits:
            split_data = load_dataset(
                conf.dataset,
                config,
                split=split,
                use_auth_token=TOKEN,
                cache_dir=conf.cache_dir,
            )
            if not conf_columns:
                # Use all string columns as text
                for feature, vtype in split_data.info.features.items():
                    if isinstance(vtype, Value) and vtype.dtype == "string":
                        conf_columns.append(feature)

            if not conf_columns:
                raise ValueError(f"No columns specified in {split}")

            clear_screen()
            logger.info(f"Using columns in {split}: {conf_columns}")

            extract_config_md5 = compute_md5(
                {
                    "config": config,
                    "split": split,
                    "columns": conf_columns,
                }
            )
            split_data = split_data.map(
                extract_text,
                fn_kwargs={"columns": conf_columns},
                num_proc=num_proc,
                cache_file_name=f"{storage_prefix}-{compute_md5(extract_config_md5)}.cache",
                load_from_cache_file=True,
                remove_columns=split_data.column_names,
                desc="Extracting text...",
            )

            if conf.embedder.name in {"SimHashEmbedder", "MinHashEmbedder"}:
                embedder: Union[SimHashEmbedder, MinHashEmbedder] = SimHashEmbedder()
                if conf.embedder.name == "MinHashEmbedder":
                    embedder = MinHashEmbedder(num_perm=conf.embedder.num_perm)

                embed_function = embedder.embed_function(
                    n_gram=conf.tokenization.ngram_size,
                    level=conf.tokenization.level,
                    use_str=conf.embedder.name == "SimHashEmbedder",
                )

                embed_config_md5 = compute_md5(
                    {
                        "config": config,
                        "split": split,
                        "columns": conf_columns,
                        "ngram_size": conf.tokenization.ngram_size,
                        "level": conf.tokenization.level,
                        "embedder": repr(embedder),
                    }
                )

                split_data = split_data.map(
                    lambda x: {
                        "__signature__": embed_function(x["__text__"]),
                    },
                    num_proc=num_proc,
                    cache_file_name=f"{storage_prefix}-{embed_config_md5}.cache",
                    load_from_cache_file=True,
                    # remove_columns=["__text__"],
                    desc="Embedding...",
                )
                split_results[split] = split_data
            else:
                embedder = SuffixArrayEmbedder(k=conf.embedder.k)  # type: ignore
                slices = embedder.embed_bash(  # type: ignore
                    split_data["__text__"],
                    skip_existing=conf.embedder.skip_existing,
                    cache_dir=conf.embedder.cache_dir,
                    temp_file_prefix=conf.embedder.temp_file_prefix,
                )
                # TODO: Speed up this part
                # slices = embedder.embed(split_data["__text__"], merge=True)
                split_results[split] = (slices, split_data["__size__"])

        clear_screen()
        records: List[Dict[str, Any]] = []

        if conf.embedder.name in {"SimHashEmbedder", "MinHashEmbedder"}:
            # All pair combinations
            for x, y in product(SPLITS, repeat=2):

                if x not in split_results or y not in split_results:
                    continue
                if SPLITS.index(x) > SPLITS.index(y):
                    continue

                clear_screen()
                logger.info(f"Looking for {y}'s duplicates in {x}")

                base_data = split_results[x]
                query_data = split_results[y]

                clustering_config_md5 = compute_md5(
                    {
                        "x": x,
                        "y": y,
                        "config": config,
                        "dataset": conf.dataset.replace("/", "_"),
                        "columns": conf_columns,
                        "ngram_size": conf.tokenization.ngram_size,
                        "level": conf.tokenization.level,
                        "embedder": repr(embedder),
                    }
                )

                clusters: List[List[int]] = (
                    simhash_clustering(
                        list(map(int, base_data["__signature__"])),
                        hamming_distance=conf.embedder.hamming_distance,
                        query_signatures=list(map(int, query_data["__signature__"])),
                        num_blocks=conf.embedder.num_blocks,
                        storage_config={
                            "type": conf.storage_config.type,
                            "redis": {
                                "host": conf.storage_config.redis.host,
                                "port": conf.storage_config.redis.port,
                            },
                            "basename": compute_md5(clustering_config_md5),
                        }
                        if conf.storage_config and conf.storage_config.type == "redis"
                        else None,
                    )
                    if conf.embedder.name == "SimHashEmbedder"
                    else lsh_clustering(
                        base_data["__signature__"],
                        threshold=conf.embedder.threshold,
                        query_signatures=query_data["__signature__"],
                        storage_config={
                            "type": conf.storage_config.type,
                            "redis": {
                                "host": conf.storage_config.redis.host,
                                "port": conf.storage_config.redis.port,
                            },
                            "basename": compute_md5(clustering_config_md5).encode("utf-8"),
                        }
                        if conf.storage_config and conf.storage_config.type == "redis"
                        else None,
                    )
                )

                duplicated_count = 0
                total_count = 0
                duplicated_size = 0
                total_size = 0
                examples = 5
                query_sizes = query_data["__size__"]

                for i, cluster in enumerate(tqdm(clusters, desc="Post-processing...")):
                    total_count += 1
                    total_size += query_sizes[i]
                    if len(cluster) <= 1:  # No duplicates
                        continue
                    duplicated_count += 1
                    duplicated_size += query_sizes[i]
                    cluster = [j for j in cluster if (x, j) != (y, i)]
                    records.append(
                        {
                            "query_index": i,
                            "query_split": y,
                            "references": cluster,
                            "reference_split": x,
                        }
                    )
                    if examples > 0:
                        table = Table(title="Examples", show_lines=True)
                        table.add_column("Query Split", justify="left", style="cyan", no_wrap=False)
                        table.add_column("Query Index", justify="left", style="cyan", no_wrap=False)
                        table.add_column("Query Instance", justify="left", style="cyan", no_wrap=False)
                        table.add_column("Duplicate Split", justify="left", style="cyan", no_wrap=False)
                        table.add_column("Duplicate Index", justify="left", style="cyan", no_wrap=False)
                        table.add_column("Duplicate", justify="left", style="magenta")
                        for ref_id, reference in zip(cluster[:10], base_data.select(cluster)["__text__"]):
                            table.add_row(
                                y,
                                str(i),
                                textwrap.shorten(query_data.select([i])["__text__"][0], width=512),
                                x,
                                str(ref_id),
                                textwrap.shorten(reference, width=512),
                            )
                        print(table)
                        examples -= 1

                duplicated_count_ratio: float = duplicated_count / total_count * 100
                duplicated_byte_ratio: float = duplicated_size / total_size * 100
                logger.info(
                    f"{x}-{y}: {duplicated_count_ratio:.2f}% ({duplicated_count}) duplicated documents, {duplicated_byte_ratio:.2f}% ({duplicated_size}) duplicated bytes",
                )

        elif conf.embedder.name == "SuffixArrayEmbedder":
            # TOOD Adopt this for all pair combinations
            duplicated_size = 0
            duplicated_count = 0
            total_count = 0
            total_size = 0
            for x in SPLITS:
                if x not in split_results:
                    continue
                slices, sizes = split_results[x]
                for i, (segments, size) in enumerate(zip(slices, sizes)):
                    total_size += size
                    total_count += 1
                    duplicated_count += 1 if segments else 0
                    duplicated_size += sum(s.stop - s.start for s in segments)
                    records.append(
                        {
                            "query_index": i,
                            "query_split": x,
                            "byte_slices": [[s.start, s.stop] for s in segments],
                        }
                    )

            duplicated_count_ratio = duplicated_count / total_count * 100
            duplicated_byte_ratio = duplicated_size / total_size * 100
            logger.info(
                f"{x}-{y}: {duplicated_count_ratio:.2f}% ({duplicated_count}) duplicated documents, {duplicated_byte_ratio:.2f}% ({duplicated_size}) duplicated bytes",
            )

        with open(f"{storage_prefix}-results.json", "w") as f:
            json.dump(records, f)

    logger.info(f"Done in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
