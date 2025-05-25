import glob
import os
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from ..flow import Flow

try:
    import orjson as json_lib

    def json_load(f):
        return json_lib.loads(f.read())

    def json_loads(b):
        return json_lib.loads(b)

    BINARY = True
except ImportError:
    import json as json_lib

    def json_load(f):
        return json_lib.load(f)

    def json_loads(b):
        return json_lib.loads(b.decode("utf-8"))

    BINARY = False


def dispatch(step) -> "Flow":
    op = step["op"]
    if op == "from_folder":
        return from_folder(step)
    elif op == "from_materialized":
        return iter(step["records"])
    elif op == "from_json":
        return from_json(step)
    elif op == "from_jsonl":
        return from_jsonl(step)
    elif op == "from_statsbomb":
        return from_statsbomb(step)
    elif op == "from_glob":
        return from_glob(step)
    elif op == "from_concat":
        return from_concat(step)
    else:
        raise ValueError(f"Unsupported source op: {step['op']}")


def from_folder(step) -> Iterator[dict]:
    """
    Create a Flow from a folder of JSON or JSONL files.

    Args:
        step (dict): A dictionary containing the path to the folder.

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    folder = step["path"]
    file_exts = (".json", ".jsonl")  # now explicitly includes .jsonl

    for filename in os.listdir(folder):
        if not filename.endswith(file_exts):
            continue

        path = os.path.join(folder, filename)

        if filename.endswith(".jsonl"):
            yield from from_jsonl({"path": path})
        elif filename.endswith(".json"):
            yield from from_json({"path": path})


def from_json(step):
    """
    Create a Flow from a JSON file.

    Args:
        step (dict): A dictionary containing the path to the JSON file.

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    path = step["path"]
    mode = "rb" if BINARY else "r"

    with open(path, mode) as f:
        data = json_load(f)
        if not isinstance(data, list):
            raise ValueError("Expected top-level list in JSON file")
        yield from data


def from_jsonl(step) -> Iterator[dict]:
    """
    Create a Flow from a JSONL file.

    Args:
        step (dict): A dictionary containing the path to the JSONL file.

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    path = step["path"]
    mode = "rb" if BINARY else "r"

    with open(path, mode) as f:
        for line in f:
            if not line.strip():
                continue
            yield json_loads(line)


def from_statsbomb(step) -> Iterator[dict]:
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


def from_glob(step) -> Iterator[dict]:
    """
    Create a Flow from a glob pattern.

    Args:
        step (dict): A dictionary containing the pattern for the glob.

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    pattern = step["pattern"]
    for path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(path):
            continue

        if path.endswith(".jsonl"):
            with open(
                path, "rb" if BINARY else "r", encoding=None if BINARY else "utf-8"
            ) as f:
                for line in f:
                    yield json_loads(line if BINARY else line.encode("utf-8"))

        elif path.endswith(".json"):
            with open(
                path, "rb" if BINARY else "r", encoding=None if BINARY else "utf-8"
            ) as f:
                data = json_load(f)
                if isinstance(data, list):
                    for r in data:
                        yield r
                elif isinstance(data, dict):
                    yield data


def from_concat(step) -> Iterator[dict]:
    """
    Create a Flow from a list of plans.

    Args:
        step (dict): A dictionary containing the plans to concatenate.

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    from ..executor import FlowExecutor

    for plan in step["plans"]:
        yield from FlowExecutor(plan).execute()
