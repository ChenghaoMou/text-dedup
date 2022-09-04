import sys
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, IterableDataset, Value
from tqdm import tqdm


def get_byte_size(x: str) -> int:  # pragma: no cover
    return sys.getsizeof(x)


def extract_text(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    text = " ".join(str(row[f]) for f in columns)
    return {"__text__": text, "__size__": get_byte_size(text)}


def dataset_map(ds, **kwargs):
    # Only apply the function to streaming dataset
    if isinstance(ds, IterableDataset):
        return ds.map(function=kwargs.pop("function"))
    return ds.map(**kwargs)


def dataset_get(ds, key: str, transform: Optional[Callable] = None):
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


def dataset_get_all_str_columns(ds: Union[IterableDataset, Dataset]) -> List[str]:
    columns = []
    for feature, vtype in ds.info.features.items():
        if isinstance(vtype, Value) and vtype.dtype == "string":
            columns.append(feature)
    return columns
