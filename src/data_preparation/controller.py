import logging

import pandas as pd

from src.data_preparation.tabular import explore, clean
from src.data_preparation.data_cleaning.vision.vision import Vision
from src.data_preparation.data_cleaning.labels.Labels import Labels

TASKS = {
    'vision': Vision,
    'labels': Labels,
}

def run_pipeline(cfg: dict, report) -> None:
    """Run pipeline based on configuration file    
    cfg (dict): configuration dict"""

    task = cfg['data']['preprocessing']
    assert task in TASKS, '[run_pipeline] task {task} does not exists'
    TASKS[task].clean(cfg, report)
    TASKS['labels'].clean(cfg, report)
    TASKS[task].data_augmentation(cfg, report)
    
