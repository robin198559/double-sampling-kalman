import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")


def log_function(func):

    def _run(*args, **kwargs):
        start_time = datetime.now()
        logging.info(f"{func.__name__}: started")
        all_results = func(*args, **kwargs)
        end_time = datetime.now()
        logging.info(f"{func.__name__}: finished using {end_time - start_time}")
        return all_results

    return _run
