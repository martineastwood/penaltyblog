"""
Tests for cloud storage glob functionality in Flow.

These tests focus on ensuring that from_glob works correctly with cloud storage paths.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from penaltyblog.matchflow import Flow
from penaltyblog.matchflow.steps.source import from_glob


def test_from_glob_with_gs_protocol():
    """Test that from_glob works correctly with gs:// protocol."""
    # Mock the fsspec filesystem and glob functionality
    mock_fs = MagicMock()
    mock_fs.glob.return_value = [
        "gs://penaltyblog/test/19739.json",
        "gs://penaltyblog/test/19740.json",
    ]
    mock_fs.isdir.return_value = False

    # Mock fsspec.filesystem to return our mock filesystem
    with patch(
        "penaltyblog.matchflow.steps.source.fsspec.filesystem", return_value=mock_fs
    ) as mock_fs_func:
        # Mock fsspec.utils.infer_storage_options
        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "gs"}

            # Mock the from_json function to avoid actual file reading
            with patch(
                "penaltyblog.matchflow.steps.source.from_json"
            ) as mock_from_json:
                mock_from_json.return_value = iter([{"id": 19739}, {"id": 19740}])

                # Test the from_glob function directly
                step = {
                    "pattern": "gs://penaltyblog/test/*.json",
                    "storage_options": {"token": "fake_token"},
                }

                results = list(from_glob(step))

                # Verify the results
                assert len(results) == 2
                assert results[0] == {"id": 19739}
                assert results[1] == {"id": 19740}

                # Verify that fsspec.filesystem was called with correct parameters
                mock_fs_func.assert_called_once_with("gs", token="fake_token")

                # Verify that glob was called with the correct pattern
                mock_fs.glob.assert_called_once_with("gs://penaltyblog/test/*.json")

                # Verify that from_json was called with correct paths and storage_options
                expected_calls = [
                    {
                        "path": "gs://penaltyblog/test/19739.json",
                        "storage_options": {"token": "fake_token"},
                    },
                    {
                        "path": "gs://penaltyblog/test/19740.json",
                        "storage_options": {"token": "fake_token"},
                    },
                ]

                assert mock_from_json.call_count == 2
                for i, call in enumerate(mock_from_json.call_args_list):
                    assert call[0][0] == expected_calls[i]


def test_from_glob_with_s3_protocol():
    """Test that from_glob works correctly with s3:// protocol."""
    # Mock the fsspec filesystem and glob functionality
    mock_fs = MagicMock()
    mock_fs.glob.return_value = ["s3://bucket/data/file1.json"]
    mock_fs.isdir.return_value = False

    with patch(
        "penaltyblog.matchflow.steps.source.fsspec.filesystem", return_value=mock_fs
    ):
        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "s3"}

            with patch(
                "penaltyblog.matchflow.steps.source.from_json"
            ) as mock_from_json:
                mock_from_json.return_value = iter([{"id": 1}])

                step = {
                    "pattern": "s3://bucket/data/*.json",
                    "storage_options": {"key": "fake_key", "secret": "fake_secret"},
                }

                results = list(from_glob(step))

                assert len(results) == 1
                assert results[0] == {"id": 1}


def test_from_glob_path_reconstruction():
    """Test that paths are correctly reconstructed when fsspec.glob returns relative paths."""
    # Mock the fsspec filesystem to return relative paths (this can happen with some cloud storage implementations)
    mock_fs = MagicMock()
    mock_fs.glob.return_value = ["19739.json", "19740.json"]  # Relative paths
    mock_fs.isdir.return_value = False

    with patch(
        "penaltyblog.matchflow.steps.source.fsspec.filesystem", return_value=mock_fs
    ):
        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "gs"}

            with patch(
                "penaltyblog.matchflow.steps.source.from_json"
            ) as mock_from_json:
                mock_from_json.return_value = iter([{"id": 19739}, {"id": 19740}])

                step = {
                    "pattern": "gs://penaltyblog/test/*.json",
                    "storage_options": {},
                }

                results = list(from_glob(step))

                # Verify that the paths were reconstructed correctly
                # Note: storage_options might not be included if empty
                expected_calls = [
                    {"path": "gs://penaltyblog/test/19739.json"},
                    {"path": "gs://penaltyblog/test/19740.json"},
                ]

                assert mock_from_json.call_count == 2
                for i, call in enumerate(mock_from_json.call_args_list):
                    actual_call = call[0][0]
                    assert actual_call["path"] == expected_calls[i]["path"]
                    # storage_options should only be included if not empty
                    if "storage_options" in actual_call:
                        assert actual_call["storage_options"] == {}


def test_from_glob_error_handling():
    """Test that from_glob handles errors correctly."""
    # Test filesystem creation error
    with patch("penaltyblog.matchflow.steps.source.fsspec.filesystem") as mock_fs_func:
        mock_fs_func.side_effect = Exception("Failed to create filesystem")

        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "gs"}

            step = {"pattern": "gs://penaltyblog/test/*.json", "storage_options": {}}

            with pytest.raises(ValueError, match="Failed to create filesystem"):
                list(from_glob(step))

    # Test glob error
    mock_fs = MagicMock()
    mock_fs.glob.side_effect = Exception("Glob failed")

    with patch(
        "penaltyblog.matchflow.steps.source.fsspec.filesystem", return_value=mock_fs
    ):
        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "gs"}

            step = {"pattern": "gs://penaltyblog/test/*.json", "storage_options": {}}

            with pytest.raises(ValueError, match="Failed to glob pattern"):
                list(from_glob(step))


def test_from_glob_local_file_paths():
    """Test that from_glob works correctly with local file paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = Path(tmpdir) / "test.json"
        with open(test_file, "w") as f:
            json.dump({"id": 123}, f)

        step = {"pattern": f"{tmpdir}/*.json", "storage_options": {}}

        results = list(from_glob(step))

        assert len(results) == 1
        assert results[0] == {"id": 123}


def test_from_glob_skips_directories():
    """Test that from_glob skips directories."""
    mock_fs = MagicMock()
    mock_fs.glob.return_value = [
        "gs://penaltyblog/test/subdir",
        "gs://penaltyblog/test/file.json",
    ]
    mock_fs.isdir.side_effect = lambda path: path.endswith(
        "subdir"
    )  # First path is a directory

    with patch(
        "penaltyblog.matchflow.steps.source.fsspec.filesystem", return_value=mock_fs
    ):
        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "gs"}

            with patch(
                "penaltyblog.matchflow.steps.source.from_json"
            ) as mock_from_json:
                mock_from_json.return_value = iter([{"id": 1}])

                step = {"pattern": "gs://penaltyblog/test/*", "storage_options": {}}

                results = list(from_glob(step))

                # Should only process the file, not the directory
                assert len(results) == 1
                assert results[0] == {"id": 1}

                # from_json should only be called once (for the file)
                assert mock_from_json.call_count == 1


def test_from_glob_with_flow_class():
    """Test that Flow.from_glob works correctly with cloud storage paths."""
    mock_fs = MagicMock()
    mock_fs.glob.return_value = ["gs://penaltyblog/test/19739.json"]
    mock_fs.isdir.return_value = False

    with patch(
        "penaltyblog.matchflow.steps.source.fsspec.filesystem", return_value=mock_fs
    ):
        with patch(
            "penaltyblog.matchflow.steps.source.fsspec.utils.infer_storage_options"
        ) as mock_infer:
            mock_infer.return_value = {"protocol": "gs"}

            with patch(
                "penaltyblog.matchflow.steps.source.from_json"
            ) as mock_from_json:
                mock_from_json.return_value = iter([{"id": 19739}])

                # Test the Flow.from_glob class method
                flow = Flow.from_glob(
                    "gs://penaltyblog/test/*.json",
                    storage_options={"token": "fake_token"},
                )

                # Verify the plan was created correctly
                assert len(flow.plan) == 1
                assert flow.plan[0]["op"] == "from_glob"
                assert flow.plan[0]["pattern"] == "gs://penaltyblog/test/*.json"
                assert flow.plan[0]["storage_options"] == {"token": "fake_token"}

                # Test that executing the flow works
                results = flow.collect()
                assert len(results) == 1
                assert results[0] == {"id": 19739}
