from dataclasses import dataclass
from typing import Type

from config import DataCfg
from data.preprocessing import (
    DB5Processor,
    DIPSProcessor,
    PreProcessor,
    SabDabProcessor,
    SinglePairProcessor,
)
from data.read_data import (
    DataReader,
    DB5Reader,
    DIPSReader,
    SabDabReader,
    SinglePairReader,
)


class DatasetError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


@dataclass
class DataProcessor:
    data_reader: DataReader
    data_processor: PreProcessor


@dataclass
class DataProcessingFactory:
    data_reader_class: Type[DataReader]
    data_processor_class: Type[PreProcessor]

    def __call__(self, data_cfg: DataCfg, lm_embed_dim: int) -> DataProcessor:
        return DataProcessor(
            self.data_reader_class.from_config(data_cfg),
            self.data_processor_class.from_config(data_cfg, lm_embed_dim, debug=False),
        )


FACTORIES = {
    "single_pair": DataProcessingFactory(SinglePairReader, SinglePairProcessor),
    "dips": DataProcessingFactory(DIPSReader, DIPSProcessor),
    "db5": DataProcessingFactory(DB5Reader, DB5Processor),
    # "sabdab": DataProcessingFactory(SabDabReader, SabDabProcessor),
}


def load_data(data_cfg: DataCfg, lm_embed_dim: int):
    """
    Split the data before processing all of it?
    """
    try:
        data_factory = FACTORIES[data_cfg.dataset]

    except KeyError as err:
        raise DatasetError(f"dataset '{data_cfg.dataset}' not found") from err

    processor = data_factory(data_cfg, lm_embed_dim)

    raw_data = processor.data_reader.load_data()
    processed_data, data_params = processor.data_processor.process_data(raw_data)

    return processed_data, data_params
