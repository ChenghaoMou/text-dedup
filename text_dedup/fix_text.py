#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse
import os
from hashlib import md5

from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import load_dataset
from text_dedup.utils import add_fix_text_args
from text_dedup.utils import add_io_args
from text_dedup.utils import add_meta_args
from text_dedup.utils.timer import Timer
import ftfy

#faster than len(t.split())
def count_spaces_iterative(text, size_to_break=5, max_chars=500):
    count = 0
    start_count = False
    chars_count = 0
    for char in text:
        if char == ' ' or char == '\n':
            if start_count:
                count += 1
                start_count = False
                if count >= size_to_break:
                    break
        else:
            start_count = True
            chars_count += 1
            if chars_count > max_chars:
                break
    return count

config = ftfy.TextFixerConfig(
    unescape_html=True,
    explain=False
)

def run_ftfy_batched(batch_examples):
    new_text = []
    for text in batch_examples['text']:
        new_text.append(ftfy.fix_text(text, config=config))
    batch_examples['text'] = new_text
    return batch_examples

def run_ftfy(example):
    example['text'] = ftfy.fix_text(example['text'], config=config)
    return example

def strip_text(example):
    example['text'] = example['text'].strip()
    return example

if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser(
        prog="text_dedup.fix_text",
        description="Fix text using ftfy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = add_io_args(parser)
    parser = add_meta_args(parser)
    parser = add_fix_text_args(parser)
    args = parser.parse_args()

    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds = load_dataset(args)

        initial_length = len(ds)
        
        if not args.not_shuffle:
            with timer("Shuffle"):
                ds = ds.add_column("__id__", list(range(len(ds))))
                ds = ds.shuffle(seed=42)

        if not args.not_run_ftfy:
            with timer("Ftfy"):
                ds = ds.map(run_ftfy, num_proc=args.num_workers, desc="Running Ftfy...")

        if not args.not_shuffle:
            with timer("Strip"):
                ds = ds.map(strip_text, num_proc=args.num_workers, desc="Stripping...")

        with timer("Filtering"):
            ds = ds.filter(lambda example: count_spaces_iterative(example['text'], size_to_break=args.min_words, max_chars=args.min_words*100) >= args.min_words, num_proc=args.num_workers)
        
        with timer("Saving"):
            if not args.not_shuffle:
                print('Sorting...')
                ds = ds.sort("__id__", kind='stable', keep_in_memory=True)
                print('Removing columns...')
                ds = ds.remove_columns("__id__")
            print('Saving...')
            ds.save_to_disk(args.output)

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {initial_length}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")
