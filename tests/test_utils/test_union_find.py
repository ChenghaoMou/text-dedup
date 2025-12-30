from text_dedup.utils.union_find import UnionFind


class TestUnionFind:
    def test_init(self) -> None:
        uf = UnionFind()
        assert uf.parent == {}
        assert uf.rank == {}

    def test_find_new_element(self) -> None:
        uf = UnionFind()
        root = uf.find(1)
        assert root == 1
        assert uf.parent[1] == 1
        assert uf.rank[1] == 0

    def test_find_existing_element(self) -> None:
        uf = UnionFind()
        uf.find(1)
        root = uf.find(1)
        assert root == 1

    def test_union_two_elements(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_union_same_element(self) -> None:
        uf = UnionFind()
        uf.union(1, 1)
        assert uf.find(1) == 1
        assert len(uf.parent) == 1

    def test_union_multiple_elements(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(4, 5)

        assert uf.find(1) == uf.find(2)
        assert uf.find(2) == uf.find(3)
        assert uf.find(1) == uf.find(3)
        assert uf.find(4) == uf.find(5)
        assert uf.find(1) != uf.find(4)

    def test_union_by_rank(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(3, 4)
        root1 = uf.find(1)
        root3 = uf.find(3)

        initial_rank1 = uf.rank[root1]
        initial_rank3 = uf.rank[root3]

        uf.union(1, 3)

        final_root = uf.find(1)
        if initial_rank1 == initial_rank3:
            assert uf.rank[final_root] == initial_rank1 + 1
        else:
            assert uf.rank[final_root] == max(initial_rank1, initial_rank3)

    def test_path_compression(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)

        root = uf.find(4)
        assert uf.parent[1] == root
        assert uf.parent[2] == root
        assert uf.parent[3] == root
        assert uf.parent[4] == root

    def test_reset(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(3, 4)
        assert len(uf.parent) > 0

        uf.reset()
        assert uf.parent == {}
        assert uf.rank == {}

    def test_get_clusters_empty(self) -> None:
        uf = UnionFind()
        clusters = uf.get_clusters()
        assert clusters == {}

    def test_get_clusters_single_element(self) -> None:
        uf = UnionFind()
        uf.find(1)
        clusters = uf.get_clusters()
        assert clusters == {1: 1}

    def test_get_clusters_no_unions(self) -> None:
        uf = UnionFind()
        uf.find(1)
        uf.find(2)
        uf.find(3)
        clusters = uf.get_clusters()

        assert clusters[1] == 1
        assert clusters[2] == 2
        assert clusters[3] == 3

    def test_get_clusters_with_unions(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(4, 5)

        clusters = uf.get_clusters()

        root_1_2_3 = uf.find(1)
        root_4_5 = uf.find(4)

        assert clusters[1] == root_1_2_3
        assert clusters[2] == root_1_2_3
        assert clusters[3] == root_1_2_3
        assert clusters[4] == root_4_5
        assert clusters[5] == root_4_5

    def test_get_clusters_complex_structure(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(5, 6)
        uf.union(2, 4)

        clusters = uf.get_clusters()

        root_1_2_3_4 = uf.find(1)
        root_5_6 = uf.find(5)

        assert clusters[1] == root_1_2_3_4
        assert clusters[2] == root_1_2_3_4
        assert clusters[3] == root_1_2_3_4
        assert clusters[4] == root_1_2_3_4
        assert clusters[5] == root_5_6
        assert clusters[6] == root_5_6

    def test_large_cluster(self) -> None:
        uf = UnionFind()
        n = 100

        for i in range(1, n):
            uf.union(i, i + 1)

        root = uf.find(1)
        for i in range(1, n + 1):
            assert uf.find(i) == root

    def test_disjoint_clusters(self) -> None:
        uf = UnionFind()

        for i in range(0, 10, 2):
            uf.union(i, i + 1)

        for i in range(0, 10, 2):
            assert uf.find(i) == uf.find(i + 1)

        for i in range(0, 8, 2):
            assert uf.find(i) != uf.find(i + 2)

    def test_union_chain(self) -> None:
        uf = UnionFind()
        elements = list(range(10))

        for i in range(len(elements) - 1):
            uf.union(elements[i], elements[i + 1])

        root = uf.find(elements[0])
        for elem in elements:
            assert uf.find(elem) == root

    def test_repeated_union(self) -> None:
        uf = UnionFind()
        uf.union(1, 2)
        root_before = uf.find(1)

        uf.union(1, 2)
        root_after = uf.find(1)

        assert root_before == root_after

    def test_mixed_operations(self) -> None:
        uf = UnionFind()

        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

        uf.union(3, 4)
        assert uf.find(3) == uf.find(4)
        assert uf.find(1) != uf.find(3)

        uf.union(2, 3)
        assert uf.find(1) == uf.find(4)

        clusters = uf.get_clusters()
        root = uf.find(1)
        assert all(clusters[i] == root for i in [1, 2, 3, 4])
