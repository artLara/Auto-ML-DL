import logging

import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data_preparation.data_cleaning.vision.ImageCleaner import ImageCleaner
from src.data_preparation.data_cleaning.vision import helpers_video
from src.data_preparation import io

INTERIM_PATH = Path('data/interim/')
class VideoCleaner():
    def __init__(self):
        pass

    def clean(self,
              cfg: dict):
        """Produce a clean files based on cgf file"""
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info('Cleaning videos...')
        file_names = io.load_file_names(cfg)
        color = cfg['data']['pipe_details']['color']
        for filename in tqdm(file_names, desc='Preprocessing frames'):
            if cfg['data']['pipe_details']['file'] == 'frames':
                frames = helpers_video.read_frames_from_dir(filename,
                                                            cfg,
                                                            color)
                
            else:
                logger.info('file type is not valid')

            
            helpers_video.write_frames_in_dir(dir=INTERIM_PATH/filename,
                                              frames=frames)
            
    
    def __process_image(self,
                        cfg: dict,
                        img: np.ndarray) -> np.ndarray:

        if 'norm' in cfg['data']['pipe_details'] and\
            cfg['data']['pipe_details']['norm']:
            img = helper.normalize(img)

        return img