from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from datasets import load_from_disk
from loguru import logger

from text_dedup.config.base import Config


def generate_report(config: Config) -> None:
    output_dir = Path(config.output.output_dir)
    dataset = load_from_disk(output_dir)
    if not isinstance(dataset, Dataset):
        raise ValueError("Dataset is not a DatasetDict")  # noqa: TRY003, TRY004
    id2cluster = dict(zip(range(len(dataset)), dataset[config.algorithm.cluster_column]))

    cluster2ids = defaultdict(list)
    for _id, cluster in id2cluster.items():
        cluster2ids[cluster].append(_id)

    top_n_clusters = sorted(cluster2ids.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    for rank, (_, ids) in enumerate(top_n_clusters):
        logger.info(f"Cluster #{rank} has {len(ids)} records")
        records = dataset.select(ids[:10])
        for record in records:
            logger.info(f"  {record[config.algorithm.text_column][:30]}...")


if __name__ == "__main__":
    from pydantic_settings import CliApp

    from text_dedup.config.base import Config

    config = CliApp.run(Config)
    generate_report(config)
