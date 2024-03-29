from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk

from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs


def load_hf_dataset(io_args: IOArgs, meta_args: MetaArgs) -> Dataset:
    """
    A simple wraper to load a huggingface dataset.

    Parameters
    ----------
    io_args : IOArgs
        The arguments for the dataset to load.
    meta_args : MetaArgs
        The arguments for the meta parameters of the dataset to load.

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
    ds = ds.map(lambda x, i: {INDEX_COLUMN: i}, with_indices=True, num_proc=io_args.num_proc)
    id2id = None
    if meta_args.idx_column is not None:
        original_index = ds[meta_args.idx_column]
        id2id = {idx: oid for idx, oid in zip(ds[INDEX_COLUMN], original_index)}
    return ds, id2id
