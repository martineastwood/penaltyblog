"""
Parallel processing for handling a streaming data pipeline, specifically the Flow class.
"""

import multiprocessing
import os
import pickle
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

if TYPE_CHECKING:
    from .flow import Flow


def _write_records(path: Path, records: list[dict]) -> None:
    from .flow import Flow

    if path.suffix.lower() == ".jsonl":
        Flow.from_records(records).to_jsonl(path)
    else:
        Flow.from_records(records).to_json_single(path)


def process_file(args) -> list[dict]:
    from .flow import Flow

    path, flow_fn, output_folder = args

    # load
    flow = (
        Flow.from_jsonl(path)
        if path.suffix.lower() == ".jsonl"
        else Flow.from_file(path)
    )

    # transform
    result = flow_fn(flow)
    if not isinstance(result, Flow):
        raise TypeError(f"`flow_fn` must return a Flow, got {type(result)} instead.")

    rows = result.collect()

    # write if needed
    if output_folder:
        out_path = Path(output_folder) / path.name
        _write_records(out_path, rows)
        return []

    return rows


def folder_flow(
    input_folder: Union[str, Path],
    flow_fn: Callable[["Flow"], "Flow"],
    output_folder: Optional[Union[str, Path]] = None,
    reduce_fn: Optional[Callable[["Flow"], "Flow"]] = None,
    n_jobs: Optional[int] = None,
    file_exts: tuple[str, ...] = (".json", ".jsonl"),
) -> Optional["Flow"]:
    """
    Apply a function to each file in a folder in parallel.

    Args:
        input_folder (Union[str, Path]): The input folder.
        flow_fn (Callable[[Flow], Flow]): The function to apply to each file.
        output_folder (Optional[Union[str, Path]], optional): The output folder. Defaults to None.
        reduce_fn (Optional[Callable[[Flow], Flow]], optional): The function to apply to the results. Defaults to None.
        n_jobs (Optional[int], optional): The number of jobs to run in parallel. Defaults to None.
        file_exts (tuple[str, ...], optional): The file extensions to process. Defaults to (".json", ".jsonl").

    Returns:
        Flow: The combined results if output_folder is None
        None: If results are written to disk
    """

    from .flow import Flow

    # check if flow_fn is pickleable, if not it will cause multiprocessing to fail
    try:
        pickle.dumps(flow_fn)
    except Exception as e:
        raise TypeError(f"flow_fn {flow_fn!r} is not pickle‐able: {e}")

    if reduce_fn is not None:
        try:
            pickle.dumps(reduce_fn)
        except Exception as e:
            raise TypeError(f"reduce_fn {reduce_fn!r} is not pickle‐able: {e}")

    input_folder = Path(input_folder)
    files = sorted(p for p in input_folder.iterdir() if p.suffix.lower() in file_exts)
    if not files:
        raise FileNotFoundError(f"No files matching {file_exts} in {input_folder}")

    # pick up all CPUs by default
    n_jobs = n_jobs or os.cpu_count() or 1

    # prepare output folder
    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    args_list = [(p, flow_fn, output_folder) for p in files]

    if n_jobs == 1:
        mapped = [process_file(arg) for arg in args_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            mapped = pool.map(process_file, args_list)

    if output_folder:
        return None

    merged = list(chain.from_iterable(mapped))
    result = Flow.from_records(merged)

    if reduce_fn:
        reduced = reduce_fn(result)
        if not isinstance(reduced, Flow):
            raise TypeError(
                f"`reduce_fn` must return a Flow, got {type(reduced)} instead."
            )
        return reduced

    return result
