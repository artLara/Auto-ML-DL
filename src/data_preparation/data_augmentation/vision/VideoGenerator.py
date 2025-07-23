import random
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.data_preparation.data_augmentation.vision.ImageGenerator import ImageGenerator
from src.data_preparation.data_cleaning.vision import helpers_video
from src.utils.io import save_csv


INTERIM_PATH = Path('data/interim/')
AUG_PATH = 'data/processed/train_aug.csv'
class VideoGenerator():
    def __init__(self):
        self.__ig = ImageGenerator()
        pass

    def generate(self,
                 cfg: dict):
        """Produce a new data based on cgf file"""
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info('Generating videos...')
        csv_path = cfg['data']['annotations']['train']
        color = cfg['data']['pipe_details']['color']
        df = pd.read_csv(csv_path)
        df_aug = df
        for _, row in tqdm(df.iterrows(),
                           total=df.shape[0],
                           desc='Data augmentation'):
            filename = row['filename']
            new_name = filename+'aug'
            new_row = row
            new_row['filename'] = new_name
            if cfg['data']['pipe_details']['file'] == 'frames':
                frames = helpers_video.read_frames_from_dir(filename,
                                                            cfg,
                                                            color)
                frames, modified = self.process_frames(frames, cfg)
                if modified:
                    df_aug.loc[len(df_aug)] = new_row
            else:
                logger.info('file type is not valid')

            
            helpers_video.write_frames_in_dir(dir=INTERIM_PATH/new_name,
                                              frames=frames)
            
        save_csv(df_aug, AUG_PATH)
        
            
    
    def process_frames(self,
                       frames: list|np.ndarray,
                       cfg: dict) -> np.ndarray:
        """Process frames
        
        Keywords arguments:
        frames (list|np.ndarray): iterable of images
        cfg(dict): dictionary with information
        
        Returns:
        frames: modified frames
        modification_flag: bool """

        flag = False
        if 'horizontal_flip' in cfg['data']['augmentation']['offline']:
            hf_prob = cfg['data']['augmentation']['offline']['horizontal_flip']
            assert 0.0 <= hf_prob <= 1.0, 'Horizontal flip ratio must be a float value between [0, 1]' 
            if hf_prob >= self.__get_prob_uniform():
                flag = True
                frames = helpers_video.flip_video(frames, 0)

        if 'vertical_flip' in cfg['data']['augmentation']['offline']:
            vf_prob = cfg['data']['augmentation']['offline']['vertical_flip']
            assert 0.0 <= vf_prob <= 1.0, 'Vertical flip ratio must be a float value between [0, 1]' 
            if vf_prob >= self.__get_prob_uniform():
                flag = True
                frames = helpers_video.flip_video(frames, 1)

        return frames, flag
    

    def __get_prob_uniform(self,
                           start: float = 0.0,
                           end: float = 1.0) -> float:
        return random.uniform(start, end)