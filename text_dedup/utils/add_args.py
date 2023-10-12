#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:16:34
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os


def add_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
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
    parser.add_argument("--cache_dir", type=str, help="`cache_dir` in load_dataset", default=".cache"),
    parser.add_argument("--revision", type=str, help="`revision` in load_dataset"),
    parser.add_argument(
        "--use_auth_token", action=argparse.BooleanOptionalAction, help="To use auth token in load_dataset from HF Hub"
    ),
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, help="Use local dataset", default=False),
    parser.add_argument("--output", type=str, help="Path to deduplicated dataset output", required=True),
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="Whether to run in debug mode", default=False
    )
    parser.add_argument(
        "--clean_cache", action=argparse.BooleanOptionalAction, help="Whether to remove all cache files", default=True
    )
    parser.add_argument(
        "--num_proc", type=int,
        help="Number of processes. Defaults to the system CPU count from os.cpu_count()",
        default=os.cpu_count()
    )
    return parser


def add_meta_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
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
        help="""Text column to use for deduplication. Concatenate desired columns beforehand if needed.""",
        required=True,
    ),
    parser.add_argument(
        "--batch_size",
        type=int,
        help="""Batch size to use for dataset iteration. Mainly for memory efficiency.
        Single-threaded dedups like exacthash especially benefit from higher batches.
        Batching process itself can take a lot of time. """,
        default=10000,
    ),
    return parser


def add_minhash_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
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
        default=5,
        help="Ngram size to use in MinHash.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum number of tokens to use in MinHash. Shorter documents will be filtered out.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed to use in MinHash")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations to use in MinHash")
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Jaccard similarity threshold to use in MinHashLSH"
    )
    parser.add_argument(
        "--b",
        type=int,
        default=None,
        help="Number of bands",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
        help="Number of rows per band",
    )
    parser.add_argument(
        "--hash_func",
        type=str,
        choices=["sha1", "xxh3"],
        default="sha1",
        help="Hashing algorithm. Defaults to sha1. xxh3 is faster",
    )
    parser.add_argument(
        "--hash_bits",
        type=int,
        choices=[16, 32, 64],
        default=64,
        help="""uint bit precision for hash. default is (np.uint)64.
        However, even when using 64bit precision, only 32 bits are extracted from hash.
        this is due to legacy reasons. refer to ekzhu/datasketch#212.
        """,
    ),

    return parser


def add_simhash_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
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
        help="""Ngram size to use in SimHash.""",
    )
    parser.add_argument("--f", type=int, default=64, choices=[64, 128], help="Simhash bit size"),
    parser.add_argument("--bit_diff", type=int, default=3, help="Bit difference to use in SimHash"),
    parser.add_argument(
        "--num_bucket", type=int, default=4, help="Number of buckets to use in SimHash, must be larger than bit_diff"
    ),

    return parser


def add_sa_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
    """
    Add Suffix Array arguments to parser. This adds the following arguments:

    - k: Minimum byte length of a duplicate substring in Suffix Array Deduplication, default 100
    - strategy: Strategy when there are overlapping duplicate substrings, default "overlapping"
        overlapping: Merge all overlapping duplicate substrings
        longest: Only keep the longest duplicate substring
    - google_repo_path: Path to google-research-deduplication codebase, required

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


def add_bloom_filter_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
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
    parser.add_argument(
        "--hash_func",
        type=str,
        choices=["md5", "sha256", "xxh3"],
        default="md5",
        help="Hash function to use in BloomFilter",
    ),
    parser.add_argument("--initial_capacity", type=int, default=100, help="Initial capacity of BloomFilter"),
    return parser


def add_exact_hash_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:  # pragma: no cover
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
    parser.add_argument(
        "--hash_func",
        type=str,
        choices=["md5", "sha256", "xxh3"],
        default="md5",
        help="Hash function to use in ExactHash. defaults to md5",
    ),
    return parser
