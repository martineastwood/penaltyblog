import os
from typing import TYPE_CHECKING, Any, Dict, Iterator, cast

if TYPE_CHECKING:
    from ..flow import Flow

import fsspec

from .source_opta import from_opta
from .source_statsbomb import from_statsbomb

try:
    import orjson as _json_lib_orjson

    BINARY = True

    def json_load(f):
        return _json_lib_orjson.loads(f.read())

    def json_loads(b):
        return _json_lib_orjson.loads(b)

except ImportError:
    import json as _json_lib_std

    BINARY = False

    def json_load(f):
        return _json_lib_std.load(f)

    def json_loads(b):
        return _json_lib_std.loads(b.decode("utf-8"))


def _handle_missing_dependency(path: str) -> None:
    """
    Check if required cloud storage dependencies are installed and provide helpful error messages.

    Args:
        path (str): The path being accessed

    Raises:
        ImportError: If required dependency is missing
    """
    protocol_mapping = {
        "s3://": "s3fs",
        "gs://": "gcsfs",
        "gcs://": "gcsfs",
        "azure://": "adlfs",
        "abfs://": "adlfs",
        "abfss://": "adlfs",
    }

    for protocol, package in protocol_mapping.items():
        if path.startswith(protocol):
            try:
                __import__(package)
            except ImportError:
                raise ImportError(
                    f"To access {protocol} paths, install {package}: pip install {package}"
                ) from None
            break


def dispatch(step) -> Iterator[Dict[Any, Any]]:
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
    elif op == "from_opta":
        return from_opta(step)
    elif op == "from_glob":
        return from_glob(step)
    elif op == "from_concat":
        return from_concat(step)
    else:
        raise ValueError(f"Unsupported source op: {step['op']}")


def from_folder(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from a folder of JSON or JSONL files.

    Args:
        step (dict): A dictionary containing:
            - path (str): The path to the folder containing the records.
            - storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    folder = step["path"]
    storage_options = step.get("storage_options", {})
    file_exts = (".json", ".jsonl")  # now explicitly includes .jsonl

    # Check dependencies for cloud storage
    _handle_missing_dependency(folder)

    # Use fsspec to list files in the directory
    fs = fsspec.filesystem(
        fsspec.utils.infer_storage_options(folder, **storage_options)["protocol"],
        **storage_options,
    )

    for filename in fs.ls(folder, detail=False):
        basename = os.path.basename(filename)
        if not basename.endswith(file_exts):
            continue

        # Create child steps with storage_options passed through
        child_step = {"path": filename}
        if storage_options:
            child_step["storage_options"] = storage_options

        if basename.endswith(".jsonl"):
            yield from from_jsonl(child_step)
        elif basename.endswith(".json"):
            yield from from_json(child_step)


def from_json(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from a JSON file.

    Args:
        step (dict): A dictionary containing:
            - path (str): The path to the JSON file.
            - storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    path = step["path"]
    storage_options = step.get("storage_options", {})

    # Check dependencies for cloud storage
    _handle_missing_dependency(path)

    mode = "rb" if BINARY else "r"

    with fsspec.open(path, mode, **storage_options) as f:
        data = json_load(f)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data
        else:
            raise ValueError("JSON file must contain either a list or a single dict")


def from_jsonl(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from a JSONL file.

    Args:
        step (dict): A dictionary containing:
            - path (str): The path to the JSONL file.
            - storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    path = step["path"]
    storage_options = step.get("storage_options", {})

    # Check dependencies for cloud storage
    _handle_missing_dependency(path)

    mode = "rb" if BINARY else "r"

    with fsspec.open(path, mode, **storage_options) as f:
        for line in f:
            if not line.strip():
                continue
            yield json_loads(line)


def from_glob(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from a glob pattern.

    Args:
        step (dict): A dictionary containing:
            - pattern (str): Glob pattern (e.g., "data/**/*.json").
            - storage_options (dict, optional): Additional options for cloud storage backends.
                For S3: {"key": "access_key", "secret": "secret_key", "endpoint_url": "url"}
                For GCS: {"token": "path/to/token.json"}
                For Azure: {"account_name": "name", "account_key": "key"}

    Returns:
        Iterator[dict]: A new Flow streaming matching files.
    """
    pattern = step["pattern"]
    storage_options = step.get("storage_options", {})

    # Check dependencies for cloud storage
    _handle_missing_dependency(pattern)

    # Use fsspec glob
    try:
        storage_options_for_fs = storage_options.copy()
        protocol = fsspec.utils.infer_storage_options(
            pattern, **storage_options_for_fs
        )["protocol"]
        fs = fsspec.filesystem(protocol, **storage_options_for_fs)
    except Exception as e:
        raise ValueError(
            f"Failed to create filesystem for pattern '{pattern}': {e}"
        ) from e

    # Get the list of files matching the pattern
    try:
        file_paths = fs.glob(pattern)
    except Exception as e:
        raise ValueError(f"Failed to glob pattern '{pattern}': {e}") from e

    for path in file_paths:
        # Skip directories
        if fs.isdir(path):
            continue

        # Handle path reconstruction for cloud storage
        if protocol != "file":
            if not path.startswith(f"{protocol}://"):
                # Extract the bucket and base path from the pattern
                pattern_parts = pattern.replace(f"{protocol}://", "").split("/", 1)
                bucket = pattern_parts[0]

                # If the path already starts with the bucket name, just prepend the protocol
                if path.startswith(f"{bucket}/"):
                    path = f"{protocol}://{path}"
                else:
                    # Otherwise, reconstruct using the full base path from the pattern
                    if "/" in pattern:
                        base_path = pattern.rsplit("/", 1)[0]
                    else:
                        base_path = pattern
                    clean_path = path.lstrip("/")
                    path = f"{base_path}/{clean_path}"
        else:
            # For local files, ensure we have absolute paths
            if not os.path.isabs(path):
                path = os.path.abspath(path)

        # Create child steps with storage_options passed through
        child_step = {"path": path}
        if storage_options:
            child_step["storage_options"] = storage_options

        if path.endswith(".jsonl"):
            yield from from_jsonl(child_step)
        elif path.endswith(".json"):
            yield from from_json(child_step)


def from_concat(step) -> Iterator[Dict[Any, Any]]:
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
