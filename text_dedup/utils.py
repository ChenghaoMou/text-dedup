#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:16:34
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse


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
you can modify it by modifying the `embed_func` function in the script
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
you can modify it by modifying the `embed_func` function in the script.
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
