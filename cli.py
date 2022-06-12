import hashlib
import logging
import os
import sys
import time
from itertools import product
from typing import Any, Dict, List

import hydra
import pandas as pd
from datasets import (Value, get_dataset_config_names, get_dataset_split_names,
                      load_dataset)
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.embedders.simhash import SimHashEmbedder
from text_dedup.embedders.suffix import SuffixArrayEmbedder
from text_dedup.utils.nn import lsh_clustering, simhash_clustering

TOKEN = os.environ.get('HF_ACCESS_TOKEN', True)


def disable_logging(library: str):
    logging.getLogger(library).setLevel(logging.ERROR)


disable_logging('simhash')
# disable_logging("datasets")

logger: logging.Logger = logging.getLogger('text_dedup')
SPLITS: List[str] = ['train', 'validation', 'test']


def get_byte_size(x: str) -> int:
    return sys.getsizeof(x)


def get_slice_text(text: str, offset: slice) -> str:
    return text.encode('utf-8')[offset].decode('utf-8', errors='ignore')


def dict_hash(dictionary: DictConfig) -> str:
    dhash = hashlib.md5()
    encoded = OmegaConf.to_yaml(dictionary).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


@hydra.main(config_name='config', config_path='configs', version_base='1.2')
def main(conf: DictConfig):
    start_time = time.time()
    conf = conf.method
    num_proc: int = conf.num_proc or os.cpu_count() or 1

    if not conf.configs:
        logger.info('No configs specified, using all available configs')
        conf.configs = get_dataset_config_names(conf.dataset, use_auth_token=TOKEN)

    for config in conf.configs:
        splits = get_dataset_split_names(conf.dataset, config, use_auth_token=TOKEN)
        if conf.embedder.name in {
            'SimHashEmbedder',
            'MinHashEmbedder',
            'SuffixArrayEmbedder',
        }:

            def serialize(x: Any) -> Any:
                if conf.embedder.name == 'SimHashEmbedder':
                    return str(x)
                return x

            def deserialize(x: Any) -> Any:
                if conf.embedder.name == 'SimHashEmbedder':
                    return int(x)
                return x

            def extract_text(row, columns) -> Dict[str, str]:

                return {'__text__': ' '.join(row[f] for f in columns)}

            split_signatures: Dict[str, Any] = {}
            conf_columns: List[str] = conf.columns or []
            for split in splits:
                split_data = load_dataset(
                    conf.dataset,
                    config,
                    split=split,
                    use_auth_token=TOKEN,
                    cache_dir=conf.cache_dir,
                )
                if not conf_columns:
                    for feature, vtype in split_data.info.features.items():
                        if isinstance(vtype, Value) and vtype.dtype == 'string':
                            conf_columns.append(feature)
                logger.info(f'Using columns in {split}: {conf_columns}')
                if conf_columns:
                    split_data = split_data.map(
                        extract_text,
                        fn_kwargs={'columns': conf_columns},
                        num_proc=num_proc,
                        cache_file_name=f"{conf.cache_dir.rstrip('/')}/{conf.dataset.replace('/', '_')}-{dict_hash(conf)}-extract.cache",
                        load_from_cache_file=True,
                        desc=f'Extracting text...',
                    )
                    if conf.embedder.name in {'SimHashEmbedder', 'MinHashEmbedder'}:
                        embedder = (
                            SimHashEmbedder()
                            if conf.embedder.name == 'SimHashEmbedder'
                            else MinHashEmbedder(num_perm=conf.embedder.num_perm)
                        )

                        embed_function = embedder.embed_function(  # type: ignore
                            n_gram=conf.tokenization.ngram_size,
                            level=conf.tokenization.level,
                        )

                        split_data = split_data.map(
                            lambda x: {
                                '__signature__': serialize(  # type: ignore
                                    embed_function(x['__text__']),
                                ),
                            },
                            num_proc=num_proc,
                            cache_file_name=f"{conf.cache_dir.rstrip('/')}/{conf.dataset.replace('/', '_')}-{dict_hash(conf)}-embed.cache",
                            load_from_cache_file=True,
                            desc=f'Embedding...',
                        )
                        split_signatures[split] = split_data

                    elif conf.embedder.name == 'SuffixArrayEmbedder':
                        embedder = SuffixArrayEmbedder(k=conf.embedder.k)
                        slices = embedder.embed_bash(
                            split_data['__text__'],
                            skip_existing=conf.embedder.skip_existing,
                            cache_dir=conf.embedder.cache_dir,
                            temp_file_prefix=conf.embedder.temp_file_prefix,
                        )
                        split_signatures[split] = (slices, split_data)
            records: List[Dict[str, Any]] = []
            if conf.embedder.name in {'SimHashEmbedder', 'MinHashEmbedder'}:
                # All pair combinations
                for x, y in product(SPLITS, repeat=2):
                    if x not in split_signatures or y not in split_signatures:
                        continue
                    if SPLITS.index(x) > SPLITS.index(y):
                        continue
                    logger.info(f'Processing {x} and {y}')
                    base_data = split_signatures[x]
                    query_data = split_signatures[y]

                    clusters: List[List[int]] = (
                        simhash_clustering(
                            list(map(deserialize, base_data['__signature__'])),
                            hamming_distance=conf.embedder.hamming_distance,
                            query_signatures=list(
                                map(deserialize, query_data['__signature__']),
                            ),
                            index_basename=f'{dict_hash(conf)}-{x}-{y}',
                            skip_indexing_if_exists=True,
                        )
                        if conf.embedder.name == 'SimHashEmbedder'
                        else lsh_clustering(
                            base_data['__signature__'],
                            threshold=conf.embedder.threshold,
                            query_signatures=query_data['__signature__'],
                            redis_basename=f'{dict_hash(conf)}-{x}-{y}',
                            redis_host='localhost',
                            redis_port=6379,
                            skip_indexing_if_exists=True,
                        )
                    )
                    duplicated_count = 0
                    total_count = 0
                    duplicated_size = 0
                    total_size = 0
                    query_docs = query_data['__text__']
                    # base_docs = base_data['__text__']
                    for i, cluster in enumerate(
                        tqdm(clusters, desc='Post-processing...'),
                    ):
                        total_count += 1
                        total_size += get_byte_size(query_docs[i])
                        if len(cluster) <= 1:
                            continue
                        duplicated_count += 1
                        duplicated_size += get_byte_size(query_docs[i])
                        cluster = [j for j in cluster if (x, j) != (y, i)]
                        records.append(
                            {
                                # 'query': query_docs[i],
                                'query_id': f'{y}-{i}',
                                # 'query_split': y,
                                'references': [
                                    {
                                        # 'ref': base_docs[j],
                                        'ref_id': f'{x}-{j}',
                                        # 'ref_split': x,
                                    }
                                    for j in cluster
                                ],
                            },
                        )
                    logger.info(
                        f'{x}-{y}: {duplicated_count / total_count * 100:.2f}% ({duplicated_count}) duplicated documents, {duplicated_size / total_size * 100:.2f}% ({duplicated_size}) duplicated bytes',
                    )

                if not records:
                    continue

            elif conf.embedder.name == 'SuffixArrayEmbedder':
                duplicated_size = 0
                duplicated_count = 0
                total_count = 0
                total_size = 0
                for x in SPLITS:
                    if x not in split_signatures:
                        continue
                    slices, base_data = split_signatures[x]
                    for i, (segments, segment_data) in enumerate(
                        zip(slices, base_data),
                    ):
                        total_size += get_byte_size(segment_data['__text__'])
                        total_count += 1
                        duplicated_count += 1 if segments else 0
                        for segment in segments:
                            duplicated_size += segment.stop - segment.start
                            records.append(
                                {
                                    # 'query': segment_data['__text__'],
                                    'query_id': f'{x}-{i}',
                                    # 'query_split': x,
                                    'substring': get_slice_text(
                                        segment_data['__text__'],
                                        segment,
                                    ),
                                },
                            )
                logger.info(
                    f'{duplicated_count / total_count * 100:.2f}% ({duplicated_count}) duplicated documents, {duplicated_size / total_size * 100:.2f}% ({duplicated_size}) duplicated bytes',
                )

            # TODO: Save the results
            pd.DataFrame(records).to_json('outputs.jsonl', lines=True, orient='records')
        else:
            logger.error(f'Unknown embedder: {conf.embedder}')

    logger.info(f'Done in {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    main()
