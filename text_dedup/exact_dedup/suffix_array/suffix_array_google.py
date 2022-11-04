from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Sequence

from text_dedup.exact_dedup.suffix_array.base import SuffixArray
from text_dedup.exact_dedup.suffix_array.utils import restore_and_merge

logger = logging.getLogger("text-dedup")


@dataclass
class GoogleSuffixArray(SuffixArray):  # pragma: no cover
    """
    A wrapper for https://github.com/google-research/deduplicate-text-datasets.

    Parameters
    ----------
    cache_dir: str
        The directory to store the temporary files.
    temp_file_prefix: str
        The prefix of the temporary files.
    google_repo_path: str
        The path to the Google repo.
    """
    cache_dir: str = os.path.join(os.getcwd(), ".cache")
    temp_file_prefix: str = "embed_temp"
    google_repo_path: str = os.path.join(os.getcwd(), "deduplicate-text-datasets")
    __prefix: str = field(init=False, repr=False)
    __base_file: str = field(init=False, repr=False)
    __query_file: str = field(init=False, repr=False)
    __offsets: List[slice] = field(init=False, repr=False)

    def __post_init__(self):
        self.cache_dir = os.path.abspath(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.__prefix: str = os.path.join(self.cache_dir, self.temp_file_prefix)
        self.__base_file: str = f"{self.__prefix}.base.{self.k}.txt"
        self.__query_file: str = f"{self.__prefix}.query.{self.k}.txt"
        self.__offsets: List[slice] = []

        logger.warning(
            "Make sure you have installed rust/cargo and initialized the submodule.",
        )

    def fit(self, data: Sequence[str]) -> GoogleSuffixArray:
        self.__offsets = self.__collect_offsets(
            self.__base_file,
            data,
        )
        self.__run_command(
            f"python scripts/make_suffix_array.py {f'{self.__prefix}.base.{self.k}.txt'}",
        )
        return self

    def predict(self, data: Sequence[str]) -> List[List[slice]]:
        base_file = self.__base_file
        query_file = self.__query_file
        query_output_file = f"{self.__prefix}.query.{self.k}.byterange"

        offsets: List[slice] = self.__collect_offsets(
            query_file,
            data,
        )
        self.__run_command(
            f"python scripts/make_suffix_array.py {query_file}",
        )
        self.__run_command(
            f"cargo run across-similar --data-file-1 {base_file} --data-file-2"
            f" {query_file} --length-threshold {self.k} --cache-dir {self.cache_dir} --num-threads {os.cpu_count()}",
        )

        self.__run_command(
            f"cargo run collect --data-file {query_file} --length-threshold {self.k}"
            f" --cache-dir {self.cache_dir} > {query_output_file}",
        )

        return restore_and_merge(
            offsets,
            query_output_file,
            self.k,
            self.merge_strategy,
        )

    def fit_predict(self, data: Sequence[str]) -> List[List[slice]]:

        self.fit(data)
        base_file = self.__base_file
        base_output_file = f"{self.__prefix}.base.{self.k}.byterange"

        self.__run_command(
            f"cargo run self-similar --data-file {base_file}"
            f" --length-threshold {self.k} --cache-dir {self.cache_dir} --num-threads {os.cpu_count()}",
        )
        self.__run_command(
            f"cargo run collect --data-file {base_file}"
            f" --length-threshold {self.k} --cache-dir {self.cache_dir} >"
            f" {base_output_file}",
        )

        return restore_and_merge(
            self.__offsets,
            base_output_file,
            self.k,
            self.merge_strategy,
        )

    @staticmethod
    def __collect_offsets(output_path: str, corpus: Sequence[str]) -> List[slice]:
        """Collect document boundaries and write everything to a file."""
        offsets: List[slice] = []
        start = 0
        with open(output_path, "wb") as f:
            for doc in corpus:
                doc_bytes = doc.encode("utf-8")
                end = start + len(doc_bytes)
                offsets.append(slice(start, end))
                start = end
                f.write(doc_bytes)
        return offsets

    def __run_command(self, cmd: str):
        p = subprocess.Popen(
            cmd,
            shell=True,
            cwd=self.google_repo_path,
        )
        code = p.wait()
        if code != 0:
            raise RuntimeError(f"Command {cmd} failed with code {code}. CWD: {self.google_repo_path}")
