import os
import logging

def create_logger(root_output_dir: str) -> None:

    # set up logger
    if not os.path.isdir(root_output_dir):
        print('creating {}'.format(root_output_dir))
        os.makedirs(root_output_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = os.path.join(root_output_dir, 'log_dir')
    if not os.path.isdir(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir, exist_ok=True)

    return logger, tensorboard_log_dir