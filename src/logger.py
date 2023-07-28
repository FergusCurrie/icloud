import logging

LOGGER_NAME = "icloud_logs"


def get_logger():
    # Logging
    logging.basicConfig(filename=f"default.log", level=logging.DEBUG)
    logger = logging.getLogger(LOGGER_NAME)

    # check if logger already has a handler
    if not logger.handlers:
        my_handler = logging.FileHandler(f"{LOGGER_NAME}.log")
        my_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        my_handler.setFormatter(formatter)
        logger.addHandler(my_handler)
    return logger
