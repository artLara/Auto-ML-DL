import random
import numpy as np
from pathlib import Path

from src.data_preparation.data_cleaning.vision import helpers_image
from src.data_preparation.data_cleaning.vision.ImageCleaner import ImageCleaner
from src.data_preparation import io

IMAGE_EXTENSION = '.jpg'
ic = ImageCleaner()
def read_frames_from_dir(dir: str,
                         cfg: dir,
                         color: str = 'rgb') -> list[np.ndarray]:
    """Load frames from directions of them
    
    Keyword arguments:
    dir (str): file path of image.
    cfg: configuration file
    color (str): rgb load image with 3 channels and
    gray just one channel. Default rgb.

    Returns np.ndarray
    """
    
    external_path = cfg['data']['annotations']['external_path']
    dir = Path(external_path+dir)
    assert dir.is_dir(), f'dir {dir} does not exists'
    img_names = io.read_file_names_from_dir(dir)
    frames = []
    for im_file in img_names:
        img = helpers_image.read_image(dir/im_file,
                                    color)
        img = ic.process_image(cfg, img) #Apply image preprocessing in other place
        frames.append(img)
    
    return frames


def write_frames_in_dir(dir: str|Path,
                        frames: list) -> None:
    """Write frames in a specific path"""
    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)
    # assert dir.exists(), f'dir {dir} does not exists'
    for index, frame in enumerate(frames):
        img_name = dir/(f'{index}{IMAGE_EXTENSION}')
        helpers_image.write_image(frame, img_name)
        

def __get_prob_uniform(start: float = 0.0,
                       end: float = 1.0) -> float:
        return random.uniform(start, end)

def flip_video(frames: list,
               flip_code: int,
               prob: float = 1.0) -> list:
    """Flip image
    Keyword arguments:
    img (np.ndarray): image to resize.
    flip_code (int): 0 horizonally, 1 vertically, -1 both.
    prob (float): probability to do it. Default 1.0, values between [0.0, 1.0]

    Return: np.ndarray
    """
    if prob >= __get_prob_uniform():
        frames = [helpers_image.flip_image(frame, flip_code) for frame in frames]

    return frames
        
    