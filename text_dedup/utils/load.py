from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk

from text_dedup.utils.args import IOArgs


def load_hf_dataset(io_args: IOArgs) -> Dataset:
    """
    A simple wraper to load a huggingface dataset.

    Parameters
    ----------
    io_args : IOArgs
        The arguments for the dataset to load.

    Returns
    -------
    Dataset
        The loaded dataset.
    """

    if io_args.local:
        ds = load_from_disk(io_args.path)
    else:
        ds = load_dataset(
            path=io_args.path,
            name=io_args.name,
            data_dir=io_args.data_dir,
            data_files=io_args.data_files,
            split=io_args.split,
            revision=io_args.revision,
            cache_dir=io_args.cache_dir,
            num_proc=io_args.num_proc,
            token=io_args.use_auth_token,
        )

    return ds
