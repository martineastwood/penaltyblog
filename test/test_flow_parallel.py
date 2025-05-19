# import json
# import pickle

# import pytest

# from penaltyblog.matchflow import Flow, folder_flow


# def filter_arsenal(flow):
#     return flow.filter(lambda r: r.get("team") == "Arsenal").select("player", "xG")


# def test_folder_flow_basic(tmp_path):
#     # Create mock input files
#     input_data = [
#         [{"team": "Arsenal", "player": "A", "xG": 0.3}],
#         [
#             {"team": "Arsenal", "player": "B", "xG": 0.5},
#             {"team": "Chelsea", "player": "C", "xG": 0.2},
#         ],
#     ]

#     for i, data in enumerate(input_data):
#         f = tmp_path / f"match_{i}.json"
#         f.write_text(json.dumps(data))

#     result = folder_flow(tmp_path, filter_arsenal, n_jobs=1)

#     expected = [
#         {"player": "A", "xG": 0.3},
#         {"player": "B", "xG": 0.5},
#     ]

#     assert isinstance(result, Flow)
#     assert result.collect() == expected


# def filter_and_select(flow: Flow) -> Flow:
#     return flow.filter(lambda r: r["team"] == "Arsenal").select("player", "xG")


# def test_folder_flow_output(tmp_path):
#     input_dir = tmp_path / "input"
#     output_dir = tmp_path / "output"
#     input_dir.mkdir()
#     output_dir.mkdir()

#     # Sample input data
#     data1 = [{"team": "Arsenal", "player": "A", "xG": 0.4}]
#     data2 = [
#         {"team": "Chelsea", "player": "B", "xG": 0.2},
#         {"team": "Arsenal", "player": "C", "xG": 0.6},
#     ]

#     # Write input files
#     (input_dir / "match1.json").write_text(json.dumps(data1))
#     (input_dir / "match2.json").write_text(json.dumps(data2))

#     # Run folder_flow with output
#     folder_flow(
#         input_dir, flow_fn=filter_and_select, output_folder=output_dir, n_jobs=1
#     )

#     # Check output files
#     out1 = json.loads((output_dir / "match1.json").read_text())
#     out2 = json.loads((output_dir / "match2.json").read_text())

#     assert out1 == [{"player": "A", "xG": 0.4}]
#     assert out2 == [{"player": "C", "xG": 0.6}]


# def filter_arsenal_for_reduce(flow: Flow) -> Flow:
#     return flow.filter(lambda r: r["team"] == "Arsenal")


# def reduce_sum_xg(flow: Flow) -> Flow:
#     return flow.summary(total_xG=("xG", "sum"))


# def test_folder_flow_with_reduce(tmp_path):
#     input_dir = tmp_path / "input"
#     input_dir.mkdir()

#     # Create input files
#     data1 = [{"team": "Arsenal", "xG": 0.3}, {"team": "Chelsea", "xG": 0.1}]
#     data2 = [{"team": "Arsenal", "xG": 0.4}, {"team": "Arsenal", "xG": 0.2}]
#     (input_dir / "a.json").write_text(json.dumps(data1))
#     (input_dir / "b.json").write_text(json.dumps(data2))

#     # Run folder_flow with reduce
#     result = folder_flow(
#         input_dir, flow_fn=filter_arsenal_for_reduce, reduce_fn=reduce_sum_xg, n_jobs=1
#     )

#     # Should return a Flow with a single row: total_xG = 0.3 + 0.4 + 0.2 = 0.9
#     output = result.collect()
#     assert isinstance(output, list)
#     assert len(output) == 1
#     assert output[0]["total_xG"] == pytest.approx(0.9)


# def test_folder_flow_requires_picklable_mapper(tmp_path):
#     # write a single JSON file
#     data = [{"foo": 42}]
#     f = tmp_path / "test.json"
#     f.write_text(json.dumps(data))

#     # lambdas are not picklable by multiprocessing.Pool
#     bad_mapper = lambda flow: flow

#     with pytest.raises(Exception) as excinfo:
#         # n_jobs=1 still uses Pool internally and will try to pickle bad_mapper
#         folder_flow(tmp_path, bad_mapper, n_jobs=1)

#     err = excinfo.value
#     # we expect either a PicklingError or a TypeError complaining about pickle
#     assert isinstance(err, (pickle.PicklingError, TypeError))
#     msg = str(err).lower()
#     assert "pickle" in msg
