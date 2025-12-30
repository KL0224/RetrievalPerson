import logging

def setup_logger(log_file="matching_log"):
    logging.basicConfig(filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
    return logging.getLogger(__name__)

logger = setup_logger()
