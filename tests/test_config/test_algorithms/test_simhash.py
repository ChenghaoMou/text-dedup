from text_dedup.config.algorithms.simhash import SimHashAlgorithmConfig


def test_permute():
    from bitarray.util import urandom

    algo = SimHashAlgorithmConfig(algorithm_name="simhash", text_column="")  # type: ignore[call-arg]
    perms = algo.create_permutations()
    data = urandom(algo.f)
    for perm in perms:
        assert perm.reverse(perm.permute(data)) == data, f"{perm.reverse(perm.permute(data))} != {data}"
