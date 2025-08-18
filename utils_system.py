import os
import logging


def create_directory(path):
    os.makedirs(path, exist_ok=True)
        

def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def set_logger(args):
    create_directory(f'{args.log_dir}/{args.type}')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename=f'{args.log_dir}/{args.type}/{args.dataset}_{args.encoder}.log')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger