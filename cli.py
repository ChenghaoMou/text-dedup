from __future__ import annotations

import os
import sys
from itertools import product

import pandas as pd
import typer
from datasets import get_dataset_split_names
from datasets import load_dataset
from datasets import Value
from rich.console import Console

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
    config: str = typer.Option(None, '--config', '-c'),
    hamming_distance: int = typer.Option(3, '--hamming-distance', '-h'),
    output: str = typer.Option('results.jsonl', '--output', '-o'),
):
    num_proc = os.cpu_count()
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
            embedder = SimHashEmbedder()
            # Python int too large to convert to C long
            split_data = split_data.map(
                lambda x: {
                    '__signature__': str(
                        embedder.embed_function(n_gram=6, level='sentencepiece')(
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
            x_embeddings, data = split_signatures[x]
            y_embeddings, reference_data = split_signatures[y]
            sizes = [sys.getsizeof(rd['__text__']) for rd in reference_data]
            clusters = simhash_clustering(
                x_embeddings,
                hamming_distance=hamming_distance,
                query_signatures=y_embeddings,
            )
            for i, cluster in enumerate(clusters):
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
            size_ratio = sum(s for c, s in zip(clusters, sizes) if len(c) > 1) / sum(
                sizes,
            )
            results[
                f'{x}-{y}'
            ] = f'{num_ratio * 100:.2f}% of docs, {size_ratio*100:.2f}% of bytes'
    df = pd.DataFrame(records)
    # df = df.assign(
    #     match=df.apply(lambda x: get_overlap(x["query"], x["ref"]), axis=1),
    # )
    # df = df.assign(match_length=df["match"].map(len))
    df.sort_values(
        ['query_split', 'ref_split'],
        ascending=[True, False], inplace=True,
    )
    console.print(results)
    console.print(df[:10])
    df.to_json(output, orient='records', lines=True)


@app.command()
def minhash_dedup(
    dataset: str = typer.Option(None, '--dataset', '-d'),
    config: str = typer.Option(None, '--config', '-c'),
    num_perm: int = typer.Option(128, '--num-perm', '-n'),
    threshold: int = typer.Option(0.9, '--threshold', '-t'),
    output: str = typer.Option('results.jsonl', '--output', '-o'),
):
    num_proc = os.cpu_count()
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
                        n_gram=6, level='sentencepiece',
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
                x_embeddings, query_signatures=y_embeddings, threshold=threshold,
            )
            for i, cluster in enumerate(clusters):
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
            size_ratio = sum(s for c, s in zip(clusters, sizes) if len(c) > 1) / sum(
                sizes,
            )
            results[
                f'{x}-{y}'
            ] = f'{num_ratio * 100:.2f}% of docs, {size_ratio*100:.2f}% of bytes'
    df = pd.DataFrame(records)
    # df = df.assign(
    #     match=df.apply(lambda x: get_overlap(x["query"], x["ref"]), axis=1),
    # )
    # df = df.assign(match_length=df["match"].map(len))
    df.sort_values(
        ['query_split', 'ref_split'],
        ascending=[True, False], inplace=True,
    )
    console.print(results)
    console.print(df[:10])
    df.to_json(output, orient='records', lines=True)


@app.command()
def suffix_dedup(
    dataset: str = typer.Option(None, '--dataset', '-d'),
    config: str = typer.Option(None, '--config', '-c'),
    k: int = typer.Option(100, '--k', '-k'),
    skip_existing: bool = typer.Option(False, '--skip-existing', '-s'),
    cache_dir: str = typer.Option('cache', '--cache-dir'),
    temp_file_prefix: str = typer.Option(
        'embed_temp', '--temp-file-prefix', '-t',
    ),
    output: str = typer.Option('results.jsonl', '--output', '-o'),
):
    def get_slice_text(text, slice):

        return text.encode('utf-8')[slice].decode('utf-8', errors='ignore')

    num_proc = os.cpu_count()
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
                                segment_data['__text__'], segment,
                            ),
                        },
                    )
    df = pd.DataFrame(records)
    console.print(f'Duplication ratio: {duplicated / total * 100:.2f}%')
    console.print(df[:10])
    df.to_json(output, orient='records', lines=True)


if __name__ == '__main__':

    app()
