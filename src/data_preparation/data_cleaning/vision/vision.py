"""
Implements differents data prepressing pipelines
for computer vision tasks"""

import logging

from src.data_preparation.data_cleaning.vision.ImageCleaner import ImageCleaner
from src.data_preparation.data_cleaning.vision.VideoCleaner import VideoCleaner

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

ic = ImageCleaner()
vc = VideoCleaner()
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
      
