from pathlib import Path
from argparse import ArgumentParser
import logging

from src.data_preparation import controller as controller_dp
# from src.feature_engineering import controller as controller_fe
# from src.model_generation import controller as controller_mg


from src import train_eval
from src.domain import config
from src.utils.Report import Report


logging.basicConfig(
    level=logging.INFO,
    format= "[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s]>> %(message)s",
    datefmt='%m/%d/%Y %I:%M:%S'
)
parser = ArgumentParser("Entrenamiento de modelos de detección de keypoints")
subparsers = parser.add_subparsers(dest="command")

parser_video_kp = subparsers.add_parser('preprocessing')
parser_video_kp.add_argument('config_file', type=str, help="yaml config file")

parser_train_kp = subparsers.add_parser('train')
parser_train_kp.add_argument('config_file', type=str, help="yaml config file")

parser_train_kp = subparsers.add_parser('test')
parser_train_kp.add_argument('config_file', type=str, help="yaml config file")

parser_train_kp = subparsers.add_parser('all')
parser_train_kp.add_argument('config_file', type=str, help="yaml config file")

# parser_img_inf = subparsers.add_parser('img_inference')
# parser_img_inf.add_argument('config_file', type=str, help="training yaml config file")
# parser_img_inf.add_argument('img_file', type=str, help="training yaml config file")

args = parser.parse_args()
cfg = config.parse_yaml_config(Path(args.config_file))
report = Report(cfg)
if args.command == 'preprocessing':
    controller_dp.run_pipeline(cfg, report)
    
elif args.command == 'test':
    test_eval.test(cfg)

elif args.command == 'all':
    print('\n\t##################### Data Preparation #####################\n')
    controller_dp.run_pipeline(cfg, report)

    

elif args.command == 'train':
    pass

else:
    print(f'Command {args.command} not found')

report.write()