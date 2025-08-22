"""
Tests for cloud storage integration in Flow.

These tests focus on the storage_options parameter and dependency checking
rather than actual cloud storage access (which would require credentials).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from penaltyblog.matchflow import Flow
from penaltyblog.matchflow.steps.source import _handle_missing_dependency


def test_storage_options_parameter_acceptance():
    """Test that all from_* methods accept storage_options parameter."""
    storage_options = {"test": "value"}

    # Test that methods accept storage_options without error
    flow_json = Flow.from_json("test.json", storage_options=storage_options)
    flow_jsonl = Flow.from_jsonl("test.jsonl", storage_options=storage_options)
    flow_folder = Flow.from_folder("test/", storage_options=storage_options)
    flow_glob = Flow.from_glob("*.json", storage_options=storage_options)

    # Check that storage_options are stored in the plan
    assert flow_json.plan[0]["storage_options"] == storage_options
    assert flow_jsonl.plan[0]["storage_options"] == storage_options
    assert flow_folder.plan[0]["storage_options"] == storage_options
    assert flow_glob.plan[0]["storage_options"] == storage_options


def test_storage_options_are_optional():
    """Test that storage_options parameter is optional."""
    # All methods should work without storage_options
    flow_json = Flow.from_json("test.json")
    flow_jsonl = Flow.from_jsonl("test.jsonl")
    flow_folder = Flow.from_folder("test/")
    flow_glob = Flow.from_glob("*.json")

    # Plans should not have storage_options key
    assert "storage_options" not in flow_json.plan[0]
    assert "storage_options" not in flow_jsonl.plan[0]
    assert "storage_options" not in flow_folder.plan[0]
    assert "storage_options" not in flow_glob.plan[0]


def test_handle_missing_dependency_s3():
    """Test dependency checking for S3 paths."""
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 's3fs'")
    ):
        with pytest.raises(
            ImportError, match="To access s3:// paths, install s3fs: pip install s3fs"
        ):
            _handle_missing_dependency("s3://bucket/file.json")


def test_handle_missing_dependency_gcs():
    """Test dependency checking for GCS paths."""
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'gcsfs'")
    ):
        with pytest.raises(
            ImportError, match="To access gs:// paths, install gcsfs: pip install gcsfs"
        ):
            _handle_missing_dependency("gs://bucket/file.json")


def test_handle_missing_dependency_azure():
    """Test dependency checking for Azure paths."""
    with patch(
        "builtins.__import__", side_effect=ImportError("No module named 'adlfs'")
    ):
        with pytest.raises(
            ImportError,
            match="To access azure:// paths, install adlfs: pip install adlfs",
        ):
            _handle_missing_dependency("azure://container/file.json")


def test_handle_missing_dependency_local_path():
    """Test that local paths don't trigger dependency checks."""
    # Should not raise any errors
    _handle_missing_dependency("/local/path/file.json")
    _handle_missing_dependency("./relative/path.json")
    _handle_missing_dependency("data.json")


def test_handle_missing_dependency_multiple_protocols():
    """Test dependency checking for various protocols."""
    test_cases = [
        ("s3://bucket/file.json", "s3fs"),
        ("gs://bucket/file.json", "gcsfs"),
        ("gcs://bucket/file.json", "gcsfs"),
        ("azure://container/file.json", "adlfs"),
        ("abfs://container/file.json", "adlfs"),
        ("abfss://container/file.json", "adlfs"),
    ]

    for path, expected_package in test_cases:
        with patch(
            "builtins.__import__",
            side_effect=ImportError(f"No module named '{expected_package}'"),
        ):
            with pytest.raises(ImportError, match=f"install {expected_package}"):
                _handle_missing_dependency(path)


def test_backwards_compatibility_with_existing_api():
    """Test that existing code without storage_options still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create test files
        json_file = base / "test.json"
        jsonl_file = base / "test.jsonl"

        json_data = [{"id": 1}, {"id": 2}]
        jsonl_data = [{"id": 3}, {"id": 4}]

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        with open(jsonl_file, "w") as f:
            for record in jsonl_data:
                f.write(json.dumps(record) + "\n")

        # Test that existing API still works
        flow_json = Flow.from_json(str(json_file))
        flow_jsonl = Flow.from_jsonl(str(jsonl_file))
        flow_folder = Flow.from_folder(str(tmpdir))
        flow_glob = Flow.from_glob(str(base / "*.json*"))

        # Verify data can be collected
        json_results = flow_json.collect()
        jsonl_results = flow_jsonl.collect()
        folder_results = flow_folder.collect()
        glob_results = flow_glob.collect()

        assert json_results == json_data
        assert jsonl_results == jsonl_data
        assert len(folder_results) == 4  # All records from both files
        assert len(glob_results) == 4  # All records from both files


def test_storage_options_integration_with_fsspec():
    """Test that storage_options are passed correctly to fsspec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "test.json"
        json_data = [{"test": "data"}]

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        # Mock fsspec.open to verify storage_options are passed
        with patch("penaltyblog.matchflow.steps.source.fsspec.open") as mock_open:
            # Configure mock to return the file content
            mock_file = Mock()
            mock_file.read.return_value = json.dumps(json_data).encode()
            mock_open.return_value.__enter__.return_value = mock_file

            # Test with storage_options
            storage_options = {"key": "value", "timeout": 30}
            flow = Flow.from_json(str(json_file), storage_options=storage_options)

            # This should trigger fsspec.open call
            try:
                results = flow.collect()
            except:
                pass  # We expect errors due to mocking, but that's OK

            # Verify fsspec.open was called with storage_options
            mock_open.assert_called_once()
            args, kwargs = mock_open.call_args
            assert args[0] == str(json_file)  # path
            assert kwargs == storage_options  # storage_options passed through


@pytest.mark.parametrize(
    "method_name,file_extension",
    [
        ("from_json", "json"),
        ("from_jsonl", "jsonl"),
    ],
)
def test_cloud_storage_method_signatures(method_name, file_extension):
    """Test that cloud storage methods have correct signatures."""
    method = getattr(Flow, method_name)

    # Should accept storage_options parameter
    flow = method(
        f"s3://bucket/file.{file_extension}", storage_options={"key": "value"}
    )
    assert flow.plan[0]["storage_options"] == {"key": "value"}

    # Should work without storage_options
    flow = method(f"s3://bucket/file.{file_extension}")
    assert "storage_options" not in flow.plan[0]


def test_folder_and_glob_methods_with_storage_options():
    """Test that from_folder and from_glob handle storage_options correctly."""
    storage_options = {"key": "test_value"}

    # Test from_folder
    flow_folder = Flow.from_folder(
        "s3://bucket/folder/", storage_options=storage_options
    )
    assert flow_folder.plan[0]["storage_options"] == storage_options

    # Test from_glob
    flow_glob = Flow.from_glob("s3://bucket/**/*.json", storage_options=storage_options)
    assert flow_glob.plan[0]["storage_options"] == storage_options


def test_mixed_local_and_cloud_paths():
    """Test that the system handles mixed local and cloud paths appropriately."""
    # Local paths should not trigger dependency checks
    _handle_missing_dependency("./local/file.json")
    _handle_missing_dependency("/absolute/local/file.json")

    # Cloud paths should trigger dependency checks
    with patch("builtins.__import__", side_effect=ImportError("No module")):
        with pytest.raises(ImportError):
            _handle_missing_dependency("s3://bucket/file.json")


def test_docstring_examples_are_valid():
    """Test that the examples in docstrings are syntactically valid."""
    # These shouldn't raise syntax errors

    # Example from from_json docstring
    storage_options_s3 = {
        "key": "access_key",
        "secret": "secret_key",
        "endpoint_url": "url",
    }
    flow1 = Flow.from_json("s3://bucket/file.json", storage_options=storage_options_s3)

    # Example from from_folder docstring
    storage_options_gcs = {"token": "path/to/token.json"}
    flow2 = Flow.from_folder("gs://bucket/folder/", storage_options=storage_options_gcs)

    # Example from from_glob docstring
    storage_options_azure = {"account_name": "name", "account_key": "key"}
    flow3 = Flow.from_glob(
        "azure://container/*.json", storage_options=storage_options_azure
    )

    # Verify the plans contain the storage_options
    assert flow1.plan[0]["storage_options"] == storage_options_s3
    assert flow2.plan[0]["storage_options"] == storage_options_gcs
    assert flow3.plan[0]["storage_options"] == storage_options_azure
