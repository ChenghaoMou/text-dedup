text-dedup
==========

Hash-based Near Deduplication Examples
--------------------------------------

MinHash + LSH
~~~~~~~~~~~~~

::

    from text_dedup.embedders.minhash import MinHashEmbedder
    from text_dedup.postprocess.clustering import lsh_clustering
    from text_dedup.postprocess.group import get_group_indices

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

    from text_dedup.embedders.simhash import SimHashEmbedder
    from text_dedup.postprocess.clustering import simhash_clustering
    from text_dedup.postprocess.group import get_group_indices

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

    from text_dedup.embedders.suffix import SuffixArrayEmbedder

    if __name__ == "__main__":

        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "This is a test",
            "This is a test",
            "This is a random test",
            "The quick brown fox and a random test"
        ]


        embedder = SuffixArrayEmbedder(k=10)
        slices = embedder.embed(corpus, merge=True, merge_strategy='longest')
        # or using the original rust code
        # slices = embedder.embed_bash(corpus)

        for sentence, intervals in zip(corpus, slices):
            print(sentence)
            print([sentence[slice] for slice in intervals])
        # The quick brown fox jumps over the lazy dog
        # ['The quick brown fox jumps over the lazy dog']
        # The quick brown fox jumps over the lazy dog
        # ['The quick brown fox jumps over the lazy dog']
        # This is a test
        # ['This is a test']
        # This is a test
        # ['This is a test']
        # This is a random test
        # ['This is a ', ' a random test']
        # The quick brown fox and a random test
        # ['The quick brown fox ', ' a random test']


Transformer Embedding Semantic Deduplication Examples
-----------------------------------------------------

::

    from text_dedup.embedders.transformer import TransformerEmbedder
    from text_dedup.postprocess.clustering import annoy_clustering
    from text_dedup.postprocess.group import get_group_indices

    if __name__ == "__main__":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        corpus = [
            "The quick brown fox jumps over the dog",
            "The quick brown fox jumps over the corgi",
            "This is a test",
            "This is a test message",
        ]

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

        embedder = TransformerEmbedder(tokenizer, model)
        embeddings = embedder.embed(corpus)

        clusters = annoy_clustering(embeddings, f=768)
        groups = get_group_indices(clusters)
        print(groups)
        # [0, 0, 2, 2]


.. toctree::
   :maxdepth: 4


   cli
   text_dedup
