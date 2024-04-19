import logging
import functools

logging.basicConfig(
    format="%(levelname)s|%(thread)d|%(message)s",
    level=logging.INFO
    )
logger = logging.getLogger(__name__)


def log(func):
    """
        Decorator function for logging.

        Args:
            - func (Callable): The function to be decorated.

        Returns:
            - Callable: The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mssg = {'function': func.__name__,'input':{'args': args, 'kwargs': kwargs}}
        try:
            output = func(*args, **kwargs)
            mssg.update({'output': output})
            logger.info(repr(mssg))
            return output
        except Exception as error:
            mssg.update({'output': error})
            logger.error(repr(mssg))
            raise error
    return wrapper