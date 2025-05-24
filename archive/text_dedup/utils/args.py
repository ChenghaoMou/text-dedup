#!/usr/bin/env python
# @Date    : 2022-11-05 09:16:34
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import functools
import os
from typing import Any

import click
from click_option_group import optgroup
from datasets.utils.version import dataclass


@dataclass
class IOArgs:
    path: str
    output: str
    name: str | None = None
    data_dir: str | None = None
    data_files: str | None = None
    split: str | None = None
    cache_dir: str = ".cache"
    revision: str | None = None
    use_auth_token: bool = False
    local: bool = False
    debug: bool = False
    clean_cache: bool = False
    num_proc: int = int(os.cpu_count() or 1)

    @staticmethod
    def option_group(func):
        @optgroup.group("Input/Output Options", help="Dataset and file handling options")
        @optgroup.option("--path", type=str, help="`path` in load_dataset", required=False)
        @optgroup.option("--name", type=str, help="`name` in load_dataset", default=None, required=False)
        @optgroup.option("--data_dir", type=str, help="`data_dir` in load_dataset", default=None)
        @optgroup.option("--data_files", type=str, help="`data_files` in load_dataset", default=None)
        @optgroup.option("--split", type=str, help="`split` in load_dataset", default=None)
        @optgroup.option("--cache_dir", type=str, help="`cache_dir` in load_dataset", default=".cache")
        @optgroup.option("--revision", type=str, help="`revision` in load_dataset", default=None)
        @optgroup.option(
            "--use_auth_token",
            help="To use auth token in load_dataset from HF Hub",
            default=False,
        )
        @optgroup.option("--local/--no-local", help="Use local dataset", default=False)
        @optgroup.option("--output", type=str, help="Path to deduplicated dataset output", required=False)
        @optgroup.option("--debug/--no-debug", help="Whether to run in debug mode", default=False)
        @optgroup.option("--clean_cache/--no-clean_cache", help="Whether to remove all cache files", default=True)
        @optgroup.option(
            "--num_proc",
            type=int,
            help="Number of processes. Defaults to the system CPU count from os.cpu_count()",
            default=os.cpu_count(),
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "io_args" not in kwargs:
                io_args = IOArgs(**{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in IOArgs.__annotations__})
            else:
                io_args = kwargs.pop("io_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in IOArgs.__annotations__}
            return func(*args, **kwargs, io_args=io_args)

        return wrapper


@dataclass
class MetaArgs:
    column: str
    batch_size: int = 10_000
    idx_column: str | None = None

    @staticmethod
    def option_group(func):
        @optgroup.group("Meta Options", help="Meta options")
        @optgroup.option("--column", type=str, help="Column to deduplicate", required=True)
        @optgroup.option("--idx_column", type=str, help="Column to index", required=False, default=None)
        @optgroup.option("--batch_size", type=int, help="Batch size for deduplication", default=10_000)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "meta_args" not in kwargs:
                meta_args = MetaArgs(**{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in MetaArgs.__annotations__})
            else:
                meta_args = kwargs.pop("meta_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in MetaArgs.__annotations__}
            return func(*args, **kwargs, meta_args=meta_args)

        return wrapper


@dataclass
class MinHashArgs:
    ngram: int = 5
    min_length: int = 5
    seed: int = 42
    num_perm: int = 250
    threshold: float = 0.7
    b: int | None = None
    r: int | None = None
    hash_func: str = "sha1"
    hash_bits: int = 64

    @staticmethod
    def option_group(func):
        @optgroup.group("Minhash Options", help="Minhash options")
        @optgroup.option("--ngram", type=int, help="Ngram size", default=5)
        @optgroup.option(
            "--min_length",
            type=int,
            help="Minimum token length of document to be considered. All but one short documents will be removed.",
            default=5,
        )
        @optgroup.option("--seed", type=int, help="Seed for Minhash", default=42)
        @optgroup.option("--num_perm", type=int, help="Number of permutations", default=250)
        @optgroup.option("--threshold", type=float, help="Threshold for Minhash", default=0.7)
        @optgroup.option("--b", type=int, help="Number of bands", default=None)
        @optgroup.option("--r", type=int, help="Number of rows per band", default=None)
        @optgroup.option(
            "--hash_func",
            help="Hashing algorithm. Defaults to sha1. xxh3 is faster",
            default="sha1",
            type=click.Choice(["sha1", "xxh3"], case_sensitive=False),
        )
        @optgroup.option(
            "--hash_bits",
            help="""uint bit precision for hash. default is (np.uint)64.
                However, even when using 64bit precision, only 32 bits are extracted from hash.
                this is due to legacy reasons. refer to ekzhu/datasketch#212.""",
            default="64",
            type=click.Choice(["32", "64", "128"], case_sensitive=False),
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "minhash_args" not in kwargs:
                minhash_args = MinHashArgs(
                    **{
                        k: (kwargs.pop(k) if k != "hash_bits" else int(kwargs.pop(k)))
                        for k in list(kwargs.keys())
                        if k in MinHashArgs.__annotations__
                    }
                )
            else:
                minhash_args = kwargs.pop("minhash_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in MinHashArgs.__annotations__}
            return func(*args, **kwargs, minhash_args=minhash_args)

        return wrapper


@dataclass
class SimHashArgs:
    ngram: int = 3
    f: int = 64
    bit_diff: int = 3
    num_bucket: int = 4

    @staticmethod
    def option_group(func):
        @optgroup.group("Simhash Options", help="Simhash options")
        @optgroup.option("--ngram", type=int, help="Ngram size", default=3)
        @optgroup.option(
            "--f", help="Simhash bit size", default="64", type=click.Choice(["64", "128"], case_sensitive=False)
        )
        @optgroup.option("--bit_diff", type=int, help="Bit difference to use in SimHash", default=3)
        @optgroup.option(
            "--num_bucket",
            type=int,
            help="Number of buckets to use in SimHash, must be larger than bit_diff",
            default=4,
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "simhash_args" not in kwargs:
                simhash_args = SimHashArgs(
                    **{
                        k: (kwargs.pop(k) if k != "f" else int(kwargs.pop(k)))
                        for k in list(kwargs.keys())
                        if k in SimHashArgs.__annotations__
                    }
                )
            else:
                simhash_args = kwargs.pop("simhash_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in SimHashArgs.__annotations__}
            return func(*args, **kwargs, simhash_args=simhash_args)

        return wrapper


@dataclass
class SAArgs:
    google_repo_path: str
    k: int = 100
    strategy: str = "overlapping"

    @staticmethod
    def option_group(func):
        @optgroup.group("Suffix Array Options", help="SA options")
        @optgroup.option(
            "--k",
            type=int,
            help="Minimum byte length of a duplicate substring in Suffix Array Deduplication",
            default=100,
        )
        @optgroup.option(
            "--strategy",
            help="Strategy when there are overlapping duplicate substrings",
            default="overlapping",
            type=click.Choice(["overlapping", "longest"], case_sensitive=False),
        )
        @optgroup.option(
            "--google_repo_path", type=str, help="Path to google-research-deduplication codebase", required=True
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "sa_args" not in kwargs:
                sa_args = SAArgs(**{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in SAArgs.__annotations__})
            else:
                sa_args = kwargs.pop("sa_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in SAArgs.__annotations__}
            return func(*args, **kwargs, sa_args=sa_args)

        return wrapper


@dataclass
class BloomFilterArgs:
    error_rate: float = 1e-6
    hash_func: str = "md5"
    initial_capacity: int = 100

    @staticmethod
    def option_group(func):
        @optgroup.group("Bloom Filter Options", help="Bloom filter options")
        @optgroup.option("--error_rate", type=float, help="Error rate to use in BloomFilter", default=1e-6)
        @optgroup.option(
            "--hash_func",
            help="Hash function to use in BloomFilte",
            default="md5",
            type=click.Choice(["md5", "sha256", "xxh3"], case_sensitive=False),
        )
        @optgroup.option("--initial_capacity", type=int, help="Initial capacity of BloomFilter", default=100)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "bloom_filter_args" not in kwargs:
                bloom_filter_args = BloomFilterArgs(
                    **{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in BloomFilterArgs.__annotations__}
                )
            else:
                bloom_filter_args = kwargs.pop("bloom_filter_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in BloomFilterArgs.__annotations__}
            return func(*args, **kwargs, bloom_filter_args=bloom_filter_args)

        return wrapper


@dataclass
class ExactHashArgs:
    hash_func: str = "md5"

    @staticmethod
    def option_group(func):
        @optgroup.group("Exact Hash Options", help="Exact hash options")
        @optgroup.option(
            "--hash_func",
            help="Hash function to use in ExactHash",
            type=click.Choice(["md5", "sha256", "xxh3"], case_sensitive=False),
            default="md5",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "exact_hash_args" not in kwargs:
                exact_hash_args = ExactHashArgs(
                    **{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in ExactHashArgs.__annotations__}
                )
            else:
                exact_hash_args = kwargs.pop("exact_hash_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in ExactHashArgs.__annotations__}
            return func(*args, **kwargs, exact_hash_args=exact_hash_args)

        return wrapper


@dataclass
class UniSimArgs:
    store_data: bool = False
    index_type: str = "approx"
    return_embeddings: bool = False
    batch_size: int = 24
    use_accelerator: bool = False
    model_id: str = "text/retsim/v1"
    index_params: dict[str, Any] | None = None
    similarity_threshold: float = 0.9
    verbose: int = 0

    @staticmethod
    def option_group(func):
        @optgroup.group("UniSim Options", help="UniSim options")
        @optgroup.option("--store_data", help="Store data in cache", is_flag=True, default=False)
        @optgroup.option("--index_type", help="Index type to use", default="approx")
        @optgroup.option("--return_embeddings", help="Return embeddings instead of scores", is_flag=True, default=False)
        @optgroup.option("--use_accelerator", help="Use accelerator for index creation", is_flag=True, default=False)
        @optgroup.option("--model_id", help="Model id to use", default="text/retsim/v1")
        @optgroup.option(
            "--index_params",
            type=dict,
            help="Index params to pass to the index creator (e.g. {'ngram': 2})",
            default=None,
        )
        @optgroup.option("--similarity_threshold", help="Similarity threshold to use", default=0.9)
        @optgroup.option("--verbose", help="Verbose level for logging", is_flag=True, default=False)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "unisim_args" not in kwargs:
                unisim_args = UniSimArgs(
                    **{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in UniSimArgs.__annotations__}
                )
            else:
                unisim_args = kwargs.pop("unisim_args")
                {kwargs.pop(k) for k in list(kwargs.keys()) if k in UniSimArgs.__annotations__}
            return func(*args, **kwargs, unisim_args=unisim_args)

        return wrapper
