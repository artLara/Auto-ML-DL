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
        img = ic.process_image(cfg, img) #Apply image preprocessing
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
        

        