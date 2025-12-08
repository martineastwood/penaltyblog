from typing import TYPE_CHECKING, Any, Dict, Iterator


def from_statsbomb(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from a StatsBomb API endpoint.

    Args:
        step (dict): A dictionary containing the source and args for the StatsBomb API endpoint.

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    if "source" not in step or "args" not in step:
        raise ValueError("from_statsbomb step must include 'source' and 'args'")

    source = step["source"]
    args = step["args"]

    try:
        import statsbombpy
    except ImportError:
        raise ImportError("Install with `pip install statsbombpy`")

    from statsbombpy import sb

    # Dispatch to the corresponding API method
    func = getattr(sb, source, None)
    if not func:
        raise ValueError(f"Unknown StatsBomb source: {source}")

    data = func(fmt="dict", **args)
    return iter(data.values())
