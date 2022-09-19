import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from datasets import Dataset
from datasets import IterableDataset
from datasets import Value
from tqdm import tqdm


def get_byte_size(x: str) -> int:
    """
    Get the byte size of a string.

    Parameters
    ----------
    x: str
        The string to be measured.

    Returns
    -------
    int
        The byte size of the string.

    Examples
    --------
    >>> get_byte_size("Hello World!")
    61
    """
    return sys.getsizeof(x)


def extract_text(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    """
    Extract the text from a record based on the columns specified.

    Parameters
    ----------
    row: Dict[str, Any]
        The record to be processed.
    columns: List[str]
        The columns to be extracted.

    Returns
    -------
    Dict[str, Any]
        The extracted text and byte size.

    Examples
    --------
    >>> extract_text({"text": "Hello World!", "id": 1}, ["text"])
    {'__text__': 'Hello World!', '__size__': 61}
    >>> extract_text({"text": "Hello World!", "id": 2}, ["text", "id"])
    {'__text__': 'Hello World! 2', '__size__': 63}
    """
    text = " ".join(str(row[f]) for f in columns)
    return {"__text__": text, "__size__": get_byte_size(text)}


def dataset_map(ds, **kwargs):  # pragma: no cover
    # Apply only the function to streaming dataset
    if isinstance(ds, IterableDataset):
        return ds.map(function=kwargs.pop("function"))
    return ds.map(**kwargs)


def dataset_get(ds, key: str, transform: Optional[Callable] = None):  # pragma: no cover
    # Return a column from a dataset
    if transform is None:
        def transform(x):
            return x

    res = []
    if isinstance(ds, IterableDataset):
        for row in tqdm(iter(ds)):
            res.append(transform(row[key]))  # type: ignore
    else:
        res = list(map(transform, ds[key]))  # type: ignore
    return res


def dataset_get_all_str_columns(ds: Union[IterableDataset, Dataset]) -> List[str]:  # pragma: no cover
    columns = []
    for feature, v_type in ds.info.features.items():
        if isinstance(v_type, Value) and v_type.dtype == "string":
            columns.append(feature)
    return columns
