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
    parser.add_argument("--cache_dir", type=str, help="`cache_dir` in load_dataset", default=".cache"),
    parser.add_argument("--revision", type=str, help="`revision` in load_dataset"),
    parser.add_argument(
        "--use_auth_token", action=argparse.BooleanOptionalAction, help="`use_auth_token` in load_dataset"
    ),
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, help="Use local dataset", default=False),
    parser.add_argument("--output", type=str, help="Path to deduplicated dataset output", required=True),
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, help="Whether to run in debug mode", default=False
    )
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
        help="""Text column to use for deduplication. Concatenate desired columns beforehand if needed.""",
        required=True,
    ),
    parser.add_argument(
        "--batch_size",
        type=int,
        help="""Batch size to use for dataset iteration. Mainly for memory efficiency.""",
        default=10000,
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
        default=5,
        help="Ngram size to use in MinHash.",
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
        help="""Ngram size to use in SimHash.""",
    )
    parser.add_argument("--f", type=int, default=64, help="Simhash bit size"),
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
