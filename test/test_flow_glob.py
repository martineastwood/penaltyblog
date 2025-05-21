import json
import tempfile
from pathlib import Path

from penaltyblog.matchflow import Flow


def write_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def write_jsonl_file(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_from_glob_reads_json_and_jsonl_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        json_file = base / "file1.json"
        jsonl_file = base / "file2.jsonl"
        subfolder = base / "sub"
        subfolder.mkdir()

        json_data = [{"id": 1}, {"id": 2}]
        jsonl_data = [{"id": 3}, {"id": 4}]
        write_json_file(json_file, json_data)
        write_jsonl_file(jsonl_file, jsonl_data)

        nested_json = subfolder / "nested.json"
        write_json_file(nested_json, [{"id": 5}])

        all_ids = {1, 2, 3, 4, 5}

        flow = Flow.from_glob(f"{tmpdir}/**/*.*json*")
        ids = {record["id"] for record in flow.collect()}

        assert ids == all_ids


def test_from_glob_ignores_non_json_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        good = base / "valid.json"
        bad = base / "ignore.txt"

        write_json_file(good, [{"id": 10}])
        bad.write_text("not json")

        flow = Flow.from_glob(f"{tmpdir}/*")
        results = flow.collect()

        assert len(results) == 1
        assert results[0]["id"] == 10


def test_from_glob_reads_single_dict_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        file = Path(tmpdir) / "record.json"
        single_record = {"id": 99}
        write_json_file(file, single_record)

        flow = Flow.from_glob(f"{tmpdir}/*.json")
        results = flow.collect()

        assert results == [single_record]
