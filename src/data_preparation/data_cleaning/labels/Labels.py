"""
Implements differents data prepressing pipelines
for labels"""

import logging

import pandas as pd

from src.data_preparation.data_cleaning.labels import helpers
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

class Labels():
      def clean(cfg: dict,
            report) -> None:
            """Clean data for differents uses.
            cfg: configuration file. Specify pipelines of preprocessing to use.

            """
            ann_df = pd.read_csv(cfg['data']['annotations']['full'])
            helpers.split_dataframe(ann_df, cfg)