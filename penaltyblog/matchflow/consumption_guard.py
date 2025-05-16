import warnings
from functools import wraps

_CONSUMED_WARNING = (
    "This Flow has already been consumed and is now empty. "
    "If you need to iterate again, call .materialize() first or create a new Flow."
)


def guard_consumption(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_is_consumable", None) and self._is_consumable():
            if getattr(self, "_consumed", False):
                warnings.warn(_CONSUMED_WARNING, RuntimeWarning, stacklevel=2)
            self._consumed = True
        return method(self, *args, **kwargs)

    return wrapper
