import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils.io import save_csv
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize(df: pd.DataFrame,
              report,
              log_level: int = 0) -> None:
    """
    Noramlization of data

    Keyword arguments:
      df(pd.Dataframe): dataframe of dataset
      report(Report): instance of Report
      log_level(int): loggin lavel

      Returns: DataFrame
    """

    scaler = MinMaxScaler() # or StandardScaler()
    columns_names = df.columns.tolist()
    data = scaler.fit_transform(df)
    df_norm = pd.DataFrame(columns=columns_names, data=data)
    if log_level > 0:
        logger.info('[Normalization] Dataset description:')
        print(df_norm.describe())

        logger.info('[Normalization] Dataset information:')
        print(df_norm.info())


    return df_norm


def df2array(df: pd.DataFrame) -> np.ndarray:
    y = df['default payment next month'].to_numpy()
    x = df.drop('default payment next month', axis=1).to_numpy()

    return x, y