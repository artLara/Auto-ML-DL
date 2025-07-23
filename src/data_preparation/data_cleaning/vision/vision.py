"""
Implements differents data prepressing pipelines
for computer vision tasks"""

import logging

from src.data_preparation.data_cleaning.vision.ImageCleaner import ImageCleaner
from src.data_preparation.data_cleaning.vision.VideoCleaner import VideoCleaner


#The follow importation should be on differents paths, modify architecture of files
from src.data_preparation.data_augmentation.vision.ImageGenerator import ImageGenerator
from src.data_preparation.data_augmentation.vision.VideoGenerator import VideoGenerator

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

ic = ImageCleaner()
vc = VideoCleaner()
ig = ImageGenerator()
vg = VideoGenerator()

class Vision():
      def clean(cfg: dict,
            report) -> None:
            """Clean data for differents uses.
            cfg: configuration file. Specify pipelines of preprocessing to use.

            """
            #Clean images
            if cfg['data']['type'] == 'images':
                  ic.clean(cfg)
            if cfg['data']['type'] == 'videos':
                  vc.clean(cfg)
      
      def data_augmentation(cfg: dict,
                            report) -> None:
            """Apply data augmentation pipelines wrote in cfg file.
            cfg: configuration file. Specify pipelines of preprocessing to use.
            """
            #Clean images
            if cfg['data']['type'] == 'images':
                  ig.generate(cfg)
            if cfg['data']['type'] == 'videos':
                  vg.generate(cfg)
