"""
Funtions to load and write files
"""
import pandas as pd
from pathlib import Path

def load_file_names(cfg: dict)->list:
    """
    Load file names into list from cfg full annotation variable. This variable
    is a path to csv file with a column named filename

    cfg(dict): configuration file
    """
    file_dir = Path(cfg['data']['annotations']['full'])
    assert file_dir.exists(), f'full annotations is not valid path: {file_dir}'
    df = pd.read_csv(file_dir)

    return df['filename'].tolist()


def read_file_names_from_dir(dir: str|Path) -> list:
    """
    Read all files from a specific path
    dir(str|Path): read file names from this direction.
    """
    dir = Path(dir)
    file_names = sorted(dir.glob('*.jpg'))
    return file_names

