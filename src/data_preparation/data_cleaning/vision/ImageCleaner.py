import logging

import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data_preparation.data_cleaning.vision import helpers_image as helper
from src.data_preparation import io

INTERIM_PATH = Path('data/interim/')
class ImageCleaner():
    def __init__(self):
        pass

    def clean(self,
              cfg: dict):
        """Produce a clean files based on cgf file"""
        #To do:
        #Call helpers functions
        #Check cfg for apply exact pipeline
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info('Cleaning images...')
        file_names = io.load_file_names(cfg)
        external_path = cfg['data']['annotations']['external_path']
        color = cfg['data']['pipe_details']['color']
        print(color)
        for filename in tqdm(file_names):
            img = helper.read_image(external_path+filename,
                                    color)
            img = self.__process_image(cfg, img)
            helper.write_image(img, INTERIM_PATH/filename)
            
    
    def process_image(self,
                      cfg: dict,
                      img: np.ndarray) -> np.ndarray:

        #Normalization should implement in feature construction module
        if 'norm' in cfg['data']['pipe_details'] and\
            cfg['data']['pipe_details']['norm']:
            img = helper.normalize(img)

        if 'resize' in cfg['data']['pipe_details']:
            target_size = cfg['data']['pipe_details']['resize']
            assert len(target_size)==3, 'Resize must be [height, width, channels]' 
            img = helper.resize_image(img,
                                      target_size[0],
                                      target_size[1],
                                      cfg['data']['pipe_details']['resize_aspect_ratio'])
            

        return img