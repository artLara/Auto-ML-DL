import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils.io import save_csv
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

def split_dataframe(df: pd.DataFrame,
                    cfg: dict) -> None:
    """
    Noramlization of data

    Keyword arguments:
      df(pd.Dataframe): dataframe of dataset
      report(Report): instance of Report

      Returns: DataFrame
    """

    train_dir = cfg['data']['annotations']['train']
    test_dir = cfg['data']['annotations']['test']
    valid_dir = cfg['data']['annotations']['test']
    train_size = int(len(df) * cfg['data']['annotations']['splits']['train'])
    test_val_size = len(df) - train_size
    train_df, tmp_df = train_test_split(df,
                                        test_size=test_val_size,
                                        random_state=42)
    val_size = int(len(df) * cfg['data']['annotations']['splits']['validation'])
    test_df = val_df = tmp_df
    if val_size > 0:
        test_df, val_df = train_test_split(tmp_df,
                                           test_size=val_size,
                                           random_state=42)


    if cfg['run']['logging_level'] > 0:
        logger.info(f'[SplitDataframe] train dir:{train_dir}')
        logger.info(f'[SplitDataframe] test dir:{test_dir}')
        logger.info(f'[SplitDataframe] val dir:{valid_dir}')
        logger.info(f'[SplitDataframe] train size:{len(train_df)}')
        print(train_df.head())
        logger.info(f'[SplitDataframe] test size:{len(test_df)}')
        print(test_df.head())
        logger.info(f'[SplitDataframe] val size:{len(val_df)}')
        print(val_df.head())


    save_csv(train_df, cfg['data']['annotations']['train'])
    save_csv(test_df, cfg['data']['annotations']['test'])
    save_csv(val_df, cfg['data']['annotations']['validation'])

