import logging
import functools

logging.basicConfig(
    format="%(levelname)s|%(thread)d|%(message)s",
    level=logging.INFO
    )
logger = logging.getLogger(__name__)


def log(func):
    """
        Log `func` inputs and outputs.

        Parameters
        ----------
        func: callable
            Function for being called.

        Returns
        -------
        wrapper: callable
            func's wrapper.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
            Take `*args` and `**kwargs` and apply `func` to them.
            Log inputs and outputs and return otuput, wether it's an expected
            value or an error.
        """
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