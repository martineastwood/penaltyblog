from typing import Any, Callable


def get_field(path: str, default: Any | None = None) -> Callable:
    """
    Safely accesses a nested field using dot notation.

    Example:
        get_field("player.name") -> lambda d: d["player"]["name"]
    """
    keys = path.split(".")

    def accessor(d):
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key, default)
        return d

    return accessor


def get_array(key: str, index: int, default: Any | None = None) -> Callable:
    """
    Safely accesses an index in an array stored at `key`.

    Example:
        get_array("location", 0) -> lambda d: d["location"][0]
    """

    def accessor(d):
        value = d.get(key, [])
        if isinstance(value, list) and len(value) > index:
            return value[index]
        return default

    return accessor


def get_statsbomb_x(key: str = "location", default: Any | None = None) -> Callable:
    """Shortcut for get_array(key, 0)"""
    return get_array(key, 0, default)


def get_statsbomb_y(key: str = "location", default: Any | None = None) -> Callable:
    """Shortcut for get_array(key, 1)"""
    return get_array(key, 1, default)
