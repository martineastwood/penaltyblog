from collections.abc import Callable
from typing import Union


class AggRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None):
        def decorator(func):
            key = name or func.__name__
            self._registry[key] = func
            return func

        return decorator

    def get_factory(self, name):
        return self._registry[name]

    def keys(self):
        return list(self._registry.keys())


def resolve_aggregator(value: Union[Callable, str, tuple], alias: str):
    """
    Resolve an aggregator definition into a callable that accepts rows.

    Args:
        value: Either a callable, a registered name, or a (name/callable, field) tuple.
        alias: The name of the output field (used for error context).

    Returns:
        Callable[[list], Any]: An aggregation function.
    """
    if callable(value):
        return value

    if isinstance(value, str):
        factory = AGGS.get_factory(value)
        return factory()

    if isinstance(value, tuple) and len(value) == 2:
        fn, field = value

        if callable(fn):
            return lambda rows: fn(rows, field)

        if isinstance(fn, str):
            factory = AGGS.get_factory(fn)
            return factory(field)

        raise TypeError(
            f"Unsupported aggregator for '{alias}': first item in tuple must be "
            f"a string or callable, got {type(fn).__name__}"
        )

    raise TypeError(
        f"Unsupported aggregator format for '{alias}': expected callable, string, "
        "or (name/callable, field) tuple, got {type(value).__name__}"
    )


AGGS = AggRegistry()
