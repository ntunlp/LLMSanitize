import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

_root_name = 'LLMSanitize'


def get_child_logger(child_name):
    return logging.getLogger(_root_name + '.' + child_name)


def setting_logger(log_file: str, local_rank: int = -1):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARNING)

    logger = logging.getLogger(_root_name)
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARNING)

    rf_handler = logging.StreamHandler(sys.stderr)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                              datefmt='%m/%d/%Y %H:%M:%S'))

    output_dir = './log_dir'
    # if local_rank not in [-1, 0]:
    #     dist.barrier()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if local_rank == 0:
    #     dist.barrier()

    if log_file:
        f_handler = logging.FileHandler(os.path.join(output_dir, log_file))
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                                 datefmt='%m/%d/%Y %H:%M:%S'))

        logger.addHandler(f_handler)

    return logger

from functools import wraps

def suspend_logging(func):
    ''' decorator to supress logging for a code block (https://stackoverflow.com/questions/7341064/disable-logging-per-method-function)
    '''
    @wraps(func)
    def inner(*args, **kwargs):
        logging.disable(logging.FATAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    return inner