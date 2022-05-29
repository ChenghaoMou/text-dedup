from __future__ import annotations

import difflib


def get_overlap(s1: str, s2: str) -> str:
    """
    Get the longest overlap between two strings.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    str
        Longest overlap.
    """
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, _, size = s.find_longest_match(0, len(s1), 0, len(s2))

    return s1[pos_a:pos_a + size]
