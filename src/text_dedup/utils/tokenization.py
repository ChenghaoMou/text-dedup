from collections.abc import Iterator
from itertools import tee


def ngrams(sequence: list[str], n: int, min_length: int = 5) -> Iterator[tuple[str, ...]]:
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    iterator
        The ngrams.
    """
    if len(sequence) < min_length:
        return []  # type: ignore[return-value]
    if len(sequence) < n:
        return [tuple(sequence)]  # type: ignore[return-value]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)
