from dataclasses import dataclass
from typing import Type

from config import DataCfg
from data.process_data import (
    DB5Processor,
    DIPSProcessor,
    PPDataSet,
    DataProcessor,
    SinglePairProcessor,
)
from data.read_data import (
    DataReader,
    CSVReader,
    DIPSReader,
    SinglePairReader,
)


class DatasetError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


@dataclass
class DataPipeline:
    data_reader: DataReader
    data_processor: DataProcessor


@dataclass
class DataPipelineFactory:
    data_reader_class: Type[DataReader]
    data_processor_class: Type[DataProcessor]

    def __call__(
        self, root_dir: str, data_cfg: DataCfg, lm_embed_dim: int
    ) -> DataPipeline:
        return DataPipeline(
            self.data_reader_class.from_config(root_dir, data_cfg),
            self.data_processor_class.from_config(
                root_dir, data_cfg, lm_embed_dim, debug=False
            ),
        )


FACTORIES = {
    "single_pair": DataPipelineFactory(SinglePairReader, SinglePairProcessor),
    "dips": DataPipelineFactory(DIPSReader, DIPSProcessor),
    "db5": DataPipelineFactory(CSVReader, DB5Processor),
    "prediction": DataPipelineFactory(CSVReader, SinglePairProcessor)
    # "sabdab": DataProcessingFactory(SabDabReader, SabDabProcessor),
}


def load_data(
    root_dir: str, data_cfg: DataCfg, lm_embed_dim: int
) -> tuple[PPDataSet, dict[str, int]]:
    """
    Split the data before processing all of it?
    """
    try:
        data_factory = FACTORIES[data_cfg.dataset]

    except KeyError as err:
        raise DatasetError(f"dataset '{data_cfg.dataset}' not found") from err

    processor = data_factory(root_dir, data_cfg, lm_embed_dim)

    raw_data = processor.data_reader.load_data()
    processed_data, data_params = processor.data_processor.process(raw_data)

    return processed_data, data_params
