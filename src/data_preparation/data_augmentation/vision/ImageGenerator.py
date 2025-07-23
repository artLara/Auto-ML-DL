import random
import logging

import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data_preparation.data_cleaning.vision import helpers_image as helper
from src.data_preparation import io

INTERIM_PATH = Path('data/interim/')
class ImageGenerator():
    def __init__(self):
        pass

    def generate(self,
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

        if 'horizontal_flip' in cfg['data']['augmentation']['offline']:
            hf_prob = cfg['data']['augmentation']['offline']['horizontal_flip']
            assert 0.0 <= hf_prob <= 1.0, 'Horizontal flip ratio must be a float value between [0, 1]' 
            img = helper.flip_image(img, 0, hf_prob) 
        
        if 'vertical_flip' in cfg['data']['augmentation']['offline']:
            vf_prob = cfg['data']['augmentation']['offline']['vertical_flip']
            assert 0.0 <= vf_prob <= 1.0, 'Vertical flip ratio must be a float value between [0, 1]' 
            img = helper.flip_image(img, 1, vf_prob) 
            

        return img
    

    