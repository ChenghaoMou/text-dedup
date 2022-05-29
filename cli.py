from __future__ import annotations

import os
import sys
from itertools import product
from typing import List

import pandas as pd
import typer
from datasets import get_dataset_config_names
from datasets import get_dataset_split_names
from datasets import load_dataset
from datasets import Value
from rich.console import Console
from tqdm import tqdm

from text_dedup.embedders.minhash import MinHashEmbedder
from text_dedup.embedders.simhash import SimHashEmbedder
from text_dedup.embedders.suffix import SuffixArrayEmbedder
from text_dedup.utils.nn import lsh_clustering
from text_dedup.utils.nn import simhash_clustering

# from text_dedup.utils.overlap import get_overlap
# from datasets import disable_progress_bar
# from datasets import get_dataset_config_names
# from datasets.utils.logging import ERROR
# from datasets.utils.logging import set_verbosity

# set_verbosity(verbosity=ERROR)
# disable_progress_bar()

console = Console()
app = typer.Typer()


@app.command()
def simhash_dedup(
    dataset: str = typer.Option(None, '--dataset', '-d'),
    configs: list[str] = typer.Option([], '--config', '-c'),
    columns: list[str] = typer.Option([], '--columns', '-f'),
    hamming_distance: int = typer.Option(3, '--hamming-distance', '-h'),
    ngram_size: int = typer.Option(6, '--ngram-size', '-n'),
    output: str = typer.Option(
        'results/simhash-{dataset}-{config}-{columns}-{ngram}-{distance}.jsonl',
        '--output',
        '-o',
    ),
):
    num_proc = os.cpu_count()
    if not configs:
        configs = get_dataset_config_names(dataset)

    for config in configs:
        splits = get_dataset_split_names(dataset, config)
        split_signatures = {}
        conf_columns = columns or []
        for split in splits:
            split_data = load_dataset(dataset, config, split=split)
            if not conf_columns:
                for feature, vtype in split_data.info.features.items():
                    if isinstance(vtype, Value) and vtype.dtype == 'string':
                        conf_columns.append(feature)
            if conf_columns:
                split_data = split_data.map(
                    lambda x: {'__text__': ' '.join(x[f] for f in conf_columns)},
                    num_proc=num_proc,
                )
                embedder = SimHashEmbedder()
                split_data = split_data.map(
                    lambda x: {
                        '__signature__': str(
                            embedder.embed_function(
                                n_gram=ngram_size, level='sentencepiece',
                            )(
                                x['__text__'],
                            ),
                        ),
                    },
                    num_proc=num_proc,
                )
                split_signatures[split] = (
                    list(map(int, split_data['__signature__'])),
                    split_data,
                )

        results = {}
        records = []
        for x, y in product(['train', 'validation', 'test'], repeat=2):
            if x not in splits or y not in splits:
                continue
            if ['train', 'validation', 'test'].index(x) > [
                'train',
                'validation',
                'test',
            ].index(y):
                continue

            if x in split_signatures and y in split_signatures:
                console.print(f'{x} vs {y}')
                x_embeddings, data = split_signatures[x]
                y_embeddings, reference_data = split_signatures[y]
                sizes = [sys.getsizeof(rd['__text__']) for rd in reference_data]
                clusters = simhash_clustering(
                    x_embeddings,
                    hamming_distance=hamming_distance,
                    query_signatures=y_embeddings,
                )
                for i, cluster in enumerate(tqdm(clusters)):
                    if len(cluster) <= 1:
                        continue
                    for j in cluster:
                        if (y, i) == (x, j):
                            continue
                        records.append(
                            {
                                'query': reference_data[i]['__text__'],
                                'query_id': f'{y}-{i}',
                                'query_split': y,
                                'ref': data[j]['__text__'],
                                'ref_id': f'{x}-{j}',
                                'ref_split': x,
                            },
                        )
                num_ratio = len(
                    [c for c in clusters if len(c) > 1],
                ) / len(y_embeddings)
                size_ratio = sum(
                    s for c, s in zip(clusters, sizes) if len(c) > 1
                ) / sum(
                    sizes,
                )
                results[
                    f'{x}-{y}'
                ] = f'{num_ratio * 100:.2f}% of docs, {size_ratio*100:.2f}% of bytes'
        if not records:
            console.print(
                f'[red]No records found for {dataset} {config} {splits} with {hamming_distance} hamming distance[/red]',
            )
            continue
        df = pd.DataFrame(records)
        df.sort_values(
            ['query_split', 'ref_split'],
            ascending=[True, False],
            inplace=True,
        )
        console.print(results)
        console.print(df[:10])

        output_filename = output.format(
            dataset=dataset,
            config=config,
            columns='_'.join(columns),
            ngram=ngram_size,
            distance=hamming_distance,
        )
        path = os.path.dirname(output_filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        df.to_json(output_filename, orient='records', lines=True)


@app.command()
def minhash_dedup(
    dataset: str = typer.Option(None, '--dataset', '-d'),
    configs: list[str] = typer.Option([], '--config', '-c'),
    columns: list[str] = typer.Option([], '--columns', '-f'),
    num_perm: int = typer.Option(128, '--num-perm', '-p'),
    threshold: float = typer.Option(0.95, '--threshold', '-t'),
    ngram_size: int = typer.Option(6, '--ngram-size', '-n'),
    output: str = typer.Option(
        'results/minhash-{dataset}-{config}-{columns}-{ngram}-{perm}-{threshold}.jsonl',
        '--output',
        '-o',
    ),
):
    num_proc = os.cpu_count()
    if not configs:
        configs = get_dataset_config_names(dataset)

    for config in configs:
        splits = get_dataset_split_names(dataset, config)
        split_signatures = {}
        for split in splits:
            split_data = load_dataset(dataset, config, split=split)
            columns = []
            for feature, vtype in split_data.info.features.items():
                if isinstance(vtype, Value) and vtype.dtype == 'string':
                    columns.append(feature)
            if columns:
                split_data = split_data.map(
                    lambda x: {'__text__': ' '.join(x[f] for f in columns)},
                    num_proc=num_proc,
                )
                embedder = MinHashEmbedder(num_perm=num_perm)
                split_data = split_data.map(
                    lambda x: {
                        '__signature__': embedder.embed_function(
                            n_gram=ngram_size,
                            level='sentencepiece',
                        )(x['__text__']),
                    },
                    num_proc=num_proc,
                )
                split_signatures[split] = (split_data['__signature__'], split_data)

        results = {}
        records = []
        for x, y in product(['train', 'validation', 'test'], repeat=2):
            if x not in splits or y not in splits:
                continue
            if ['train', 'validation', 'test'].index(x) > [
                'train',
                'validation',
                'test',
            ].index(y):
                continue

            if x in split_signatures and y in split_signatures:
                x_embeddings, data = split_signatures[x]
                y_embeddings, reference_data = split_signatures[y]
                sizes = [sys.getsizeof(rd['__text__']) for rd in reference_data]
                clusters = lsh_clustering(
                    x_embeddings,
                    query_signatures=y_embeddings,
                    threshold=threshold,
                )
                for i, cluster in enumerate(tqdm(clusters, desc='Exporting...')):
                    if len(cluster) <= 1:
                        continue
                    for j in cluster:
                        if (y, i) == (x, j):
                            continue
                        records.append(
                            {
                                'query': reference_data[i]['__text__'],
                                'query_id': f'{y}-{i}',
                                'query_split': y,
                                'ref': data[j]['__text__'],
                                'ref_id': f'{x}-{j}',
                                'ref_split': x,
                            },
                        )
                num_ratio = len(
                    [c for c in clusters if len(c) > 1],
                ) / len(y_embeddings)
                size_ratio = sum(
                    s for c, s in zip(clusters, sizes) if len(c) > 1
                ) / sum(
                    sizes,
                )
                results[
                    f'{x}-{y}'
                ] = f'{num_ratio * 100:.2f}% of docs, {size_ratio*100:.2f}% of bytes'
        df = pd.DataFrame(records)
        df.sort_values(
            ['query_split', 'ref_split'],
            ascending=[True, False],
            inplace=True,
        )
        console.print(results)
        console.print(df[:10])
        output_filename = output.format(
            dataset=dataset,
            config=config,
            columns='_'.join(columns),
            ngram=ngram_size,
            threshold=int(threshold * 100),
            perm=num_perm,
        )
        path = os.path.dirname(output_filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        df.to_json(output, orient='records', lines=True)


@app.command()
def suffix_dedup(
    dataset: str = typer.Option(None, '--dataset', '-d'),
    configs: list[str] = typer.Option([], '--config', '-c'),
    columns: list[str] = typer.Option([], '--columns', '-f'),
    k: int = typer.Option(100, '--k', '-k'),
    skip_existing: bool = typer.Option(False, '--skip-existing', '-s'),
    cache_dir: str = typer.Option('cache', '--cache-dir'),
    temp_file_prefix: str = typer.Option(
        'embed_temp',
        '--temp-file-prefix',
        '-t',
    ),
    output: str = typer.Option(
        'results/suffix-{dataset}-{config}-{columns}-{k}.jsonl', '--output', '-o',
    ),
):
    def get_slice_text(text, offset: slice):

        return text.encode('utf-8')[offset].decode('utf-8', errors='ignore')

    num_proc = os.cpu_count()
    if not configs:
        configs = get_dataset_config_names(dataset)

    for config in configs:
        splits = get_dataset_split_names(dataset, config)
        split_signatures = {}
        for split in splits:
            split_data = load_dataset(dataset, config, split=split)
            columns = []
            for feature, vtype in split_data.info.features.items():
                if isinstance(vtype, Value) and vtype.dtype == 'string':
                    columns.append(feature)
            if columns:
                split_data = split_data.map(
                    lambda x: {'__text__': ' '.join(x[f] for f in columns)},
                    num_proc=num_proc,
                )
                embedder = SuffixArrayEmbedder(k=k)
                slices = embedder.embed_bash(
                    split_data['__text__'],
                    skip_existing=skip_existing,
                    cache_dir=cache_dir,
                    temp_file_prefix=temp_file_prefix,
                )
                split_signatures[split] = (slices, split_data)

        records = []
        total = 0
        duplicated = 0
        for x in ['train', 'validation', 'test']:
            if x not in splits:
                continue
            if x in split_signatures:
                slices, data = split_signatures[x]
                for i, (segments, segment_data) in enumerate(zip(slices, data)):
                    total += len(segment_data['__text__'].encode('utf-8'))
                    for segment in segments:
                        duplicated += segment.stop - segment.start
                        records.append(
                            {
                                'query': segment_data['__text__'],
                                'query_id': f'{x}-{i}',
                                'query_split': x,
                                'substring': get_slice_text(
                                    segment_data['__text__'],
                                    segment,
                                ),
                            },
                        )
        df = pd.DataFrame(records)
        console.print(f'Duplication ratio: {duplicated / total * 100:.2f}%')
        console.print(df[:10])
        output_filename = output.format(
            dataset=dataset,
            config=config,
            columns='_'.join(columns),
            k=k,
        )
        path = os.path.dirname(output_filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        df.to_json(output, orient='records', lines=True)


if __name__ == '__main__':

    app()
