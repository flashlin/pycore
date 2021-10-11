import logging
import datetime
import traceback
import wrapt
from asq.initiators import query

from common.io import get_file_list_by_pattern, remove_files

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(levelname)s %(asctime)s %(module)s:%(lineno)d] %(message)s',
    datefmt='%Y%m%d %H:%M:%S')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

log_filename = datetime.datetime.now().strftime("logs/%Y-%m-%d_%H_%M_%S.log")
fh = logging.FileHandler(log_filename, encoding='utf-8')
# fh.setLevel(logging.DEBUG)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# logger.addHandler(ch)
logger.addHandler(fh)

log_files = sorted(get_file_list_by_pattern("logs", ".*"), reverse=True)
old_log_files = query(log_files).skip(3).to_list()
remove_files(old_log_files)


@wrapt.decorator
def log(func, instance, args, kwargs):
    func_name = func.__name__
    enter_message = f"{func_name} args:{args} kwargs:{kwargs}"
    logger.info(enter_message)
    try:
        return func(*args, **kwargs)
    except BaseException:
        error_message = traceback.format_exc()
        logger.error(f"{error_message}")


if __name__ == "__main__":
    logging.debug('debug')
    logging.info('info')
    logging.warning('warning')
    logging.error('error')
    logging.critical('critical')
