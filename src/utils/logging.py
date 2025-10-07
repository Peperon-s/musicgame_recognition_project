import logging, os

def get_logger(name="app", log_dir="project/logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)