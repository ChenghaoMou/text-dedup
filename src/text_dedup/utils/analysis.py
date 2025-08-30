#
# created     : Thu May 22 2025
# author      : Chenghao Mou <mouchenghao at gmail dot com>
# license     : Apache-2.0
# description : Compute the optimal parameters for MinHashLSH.
#

# pyright: reportAny=false

from scipy.integrate import quad as integrate


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> tuple[int, int]:
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of false positive and false negative areas, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.
    """

    def false_positive_area(threshold: float, b: int, r: int) -> float:
        """Source: `datasketch.lsh`"""

        def proba(s: float) -> float:
            return float(1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, 0.0, threshold)
        return float(a)

    def false_negative_area(threshold: float, b: int, r: int) -> float:
        """Source: `datasketch.lsh`"""

        def proba(s: float) -> float:
            return float(1 - (1 - (1 - s ** float(r)) ** float(b)))

        a, _ = integrate(proba, threshold, 1.0)
        return float(a)

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt
