#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:16:34
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse
import os
from typing import List, Set

import datasets
import networkit as nk
from tqdm import tqdm


def add_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add input/output arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument("--path", type=str, help="`path` in load_dataset", required=True),
    parser.add_argument("--name", type=str, help="`name` in load_dataset"),
    parser.add_argument("--data_dir", type=str, help="`data_dir` in load_dataset"),
    parser.add_argument("--data_files", type=str, help="`data_files` in load_dataset"),
    parser.add_argument("--split", type=str, help="`split` in load_dataset"),
    parser.add_argument("--cache_dir", type=str, help="`cache_dir` in load_dataset"),
    parser.add_argument(
        "--use_auth_token", action=argparse.BooleanOptionalAction, help="`use_auth_token` in load_dataset"
    ),
    parser.add_argument("--output_dir", type=str, help="Path to save all results", required=True),
    parser.add_argument("--index_name", type=str, help="Name of the index file, will be saved in `output_dir`"),
    parser.add_argument(
        "--reuse_index", action=argparse.BooleanOptionalAction, help="Reuse the index if already exists"
    ),
    parser.add_argument("--graph_name", type=str, help="Name of the cluster file, will be saved in `output_dir`"),
    parser.add_argument(
        "--reuse_graph", action=argparse.BooleanOptionalAction, help="Reuse the cluster if already exists"
    ),
    parser.add_argument(
        "--dedup_name", type=str, help="Name of the deduplicated dataset, will be saved in `output_dir`", required=True
    ),
    return parser


def add_meta_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add meta arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--column",
        type=str,
        help="""
Text column to use for deduplication. If multiple columns are desired,
please concatenate them into one column before using this script
    """,
        required=True,
    ),
    return parser


def add_minhash_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add MinHash arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--ngram",
        type=int,
        default=3,
        help="""
Ngram size to use in MinHash. The tokenization is space-based,
you can modify it by modifying the `ngrams` function in `utils.py`
    """,
    ),
    parser.add_argument("--seed", type=int, default=42, help="Seed to use in MinHash"),
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations to use in MinHash"),
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Jaccard similarity threshold to use in MinHashLSH"
    ),
    return parser


def add_simhash_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add SimHash arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--ngram",
        type=int,
        default=3,
        help="""
Ngram size to use in SimHash. The tokenization is space-based,
you can modify it by modifying the `ngrams` function in `utils.py`
    """,
    )
    parser.add_argument("--bit_diff", type=int, default=3, help="Bit difference to use in SimHash"),
    parser.add_argument(
        "--num_bucket", type=int, default=4, help="Number of buckets to use in SimHash, must be larger than bit_diff"
    ),
    return parser


def add_sa_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add Suffix Array arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--k", type=int, default=100, help="Minimum byte length of a duplicate substring in Suffix Array Deduplication"
    ),
    parser.add_argument(
        "--strategy",
        type=str,
        default="overlapping",
        help="Strategy when there are overlapping duplicate substrings",
        choices=["overlapping", "longest"],
    )
    parser.add_argument(
        "--google_repo_path", type=str, help="Path to google-research-deduplication codebase", required=True
    ),
    return parser


def add_bloom_filter_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add Bloom Filter arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument("--error_rate", type=float, default=1e-6, help="Error rate to use in BloomFilter"),
    parser.add_argument("--hash_func", type=str, default="md5", help="Hash function to use in BloomFilter"),
    parser.add_argument("--initial_capacity", type=int, default=100, help="Initial capacity of BloomFilter"),
    return parser


def add_exact_hash_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add Exact Hash arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument("--hash_func", type=str, default="md5", help="Hash function to use in ExactHash"),
    return parser


def ngrams(content: str, ngram: int = 1) -> List[str]:
    """
    Generate ngrams from a string.

    Parameters
    ----------
    content : str
        The text to generate ngrams from.
    ngram : int
        The size of the ngrams.

    Returns
    -------
    ngrams : List[str]
        List of ngrams.

    Examples
    --------
    >>> ngrams("hello world!", ngram=2)
    ['hello world!']
    >>> ngrams("hello world!", ngram=1)
    ['hello', 'world!']
    >>> ngrams("This is a test message", ngram=3)
    ['This is a', 'test message']
    >>> ngrams("This is a test message", ngram=2)
    ['This is', 'a test', 'message']
    """
    tokens = [t for t in content.split(" ") if t]
    ngrams = [" ".join(tokens[i : i + ngram]) for i in range(0, len(tokens), ngram)]
    return ngrams


def find_duplicate_components(
    records: datasets,
    input_graph: str | None = None,
    output_graph: str | None = None,
) -> Set[int]:
    """
    Find the duplicate components in a graph.

    Parameters
    ----------
    records : Iterable | Dataset
        The dataset that contains the neighbors.
    input_graph : str | None, optional
        The path to the input graph, by default None
    output_graph : str | None, optional
        The path to the output graph, by default None

    Returns
    -------
    Set[int]
        The set of duplicate components.

    Examples
    --------
    >>> records = [{"__id__": 0, "__neighbors__": [1]}, {"__id__": 1, "__neighbors__": [0]}]
    >>> find_duplicate_components(records)
    {1}
    """
    if input_graph is not None:
        g = nk.readGraph(str(input_graph), nk.Format.NetworkitBinary)
    else:
        g = nk.graph.Graph()
        for record in tqdm(records, desc="Constructing graph..."):
            for y in record["__neighbors__"]:
                g.addEdge(record["__id__"], y, addMissing=True)

        if output_graph is not None:
            if os.path.exists(output_graph):
                os.remove(output_graph)
            nk.writeGraph(g, str(output_graph), nk.Format.NetworkitBinary)

    to_remove: Set[int] = set()
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    for component in tqdm(cc.getComponents(), desc="Iterating over components..."):
        component = sorted(component)
        to_remove.update(component[1:])

    return to_remove
