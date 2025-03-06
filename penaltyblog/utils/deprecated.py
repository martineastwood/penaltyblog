import functools
import warnings


def deprecated(reason=None):
    """
    Decorator to mark functions as deprecated. It will raise a DeprecationWarning
    when the function is called.

    Parameters:
        reason (str): Optional message explaining why it’s deprecated
                      and what to use instead.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated."
            if reason:
                message += f" {reason}"
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
