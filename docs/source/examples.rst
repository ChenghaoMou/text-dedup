Examples
==========

Hash-based Near Deduplication Examples
--------------------------------------

MinHash + LSH
~~~~~~~~~~~~~

::

    from text_dedup.near_dedup import MinHashEmbedder
    from text_dedup.postprocess import lsh_clustering
    from text_dedup.postprocess import get_group_indices

    if __name__ == "__main__":
        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "This is a test",
            "This is a test",
        ]

        embedder = MinHashEmbedder()
        embeddings = embedder.embed(corpus)

        clusters = lsh_clustering(embeddings)
        groups = get_group_indices(clusters)
        print(groups)
        # [0, 0, 2, 2]

SimHash
~~~~~~~

::

    from text_dedup.near_dedup import SimHashEmbedder
    from text_dedup.postprocess import simhash_clustering
    from text_dedup.postprocess import get_group_indices

    if __name__ == "__main__":
        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "This is a test",
            "This is a test",
        ]

        embedder = SimHashEmbedder()
        embeddings = embedder.embed(corpus)

        clusters = simhash_clustering(embeddings)
        groups = get_group_indices(clusters)
        print(groups)
        # [0, 0, 2, 2]


Suffix Array Substring Exact Deduplication Examples
---------------------------------------------------

::

    from text_dedup.exact_dedup import PythonSuffixArrayDeduplicator

    if __name__ == "__main__":
        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "This is a test",
            "This is a test",
            "This is a random test",
            "The quick brown fox and a random test"
        ]

        deduplicator = PythonSuffixArrayDeduplicator(k=10, merge_strategy='overlapping')
        slices = deduplicator.fit_predict(corpus)
        for sentence, intervals in zip(corpus, slices):
            print(sentence)
            print([sentence.encode('utf-8')[s].decode('utf-8', errors='ignore') for s in intervals])
        # The quick brown fox jumps over the lazy dog
        # ['The quick brown fox jumps over the lazy dog']
        # The quick brown fox jumps over the lazy dog
        # ['The quick brown fox jumps over the lazy dog']
        # This is a test
        # ['This is a test']
        # This is a test
        # ['This is a test']
        # This is a random test
        # ['This is a random test']
        # The quick brown fox and a random test
        # ['The quick brown fox ', ' a random test']


Transformer Embedding Semantic Deduplication Examples
-----------------------------------------------------

::

    from text_dedup.semantic_dedup import TransformerEmbedder
    from text_dedup.postprocess import annoy_clustering
    from text_dedup.postprocess import get_group_indices

    if __name__ == "__main__":

        corpus = [
            "The quick brown fox jumps over the dog",
            "The quick brown fox jumps over the corgi",
            "This is a test",
            "This is a test message",
        ]

        embedder = TransformerEmbedder("bert-base-uncased")
        embeddings = embedder.embed(corpus)

        clusters = annoy_clustering(embeddings, f=768)
        groups = get_group_indices(clusters)
        print(groups)
        # [0, 0, 2, 2]
