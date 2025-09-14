==================================
Working with Files: Input & Output
==================================

Flow makes it easy to **load, stream, and save structured JSON data** from a variety of sources. Whether you're pulling from disk, an API, or a folder of ``.jsonl`` files — Flow provides a consistent, lazy interface for building pipelines.

📥 Loading Data into Flow
=========================

Use ``Flow.from_*`` methods to create a new Flow from Python objects or files.

🧠 From Python Data: ``.from_records(...)``
-------------------------------------------

.. code-block:: python

   from penaltyblog.matchflow import Flow

   data = [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]
   flow = Flow.from_records(data)

Also works with single dicts or generators:

.. code-block:: python

   flow = Flow.from_records({"id": 3, "value": "C"})

   def gen():
       for i in range(3):
           yield {"id": i}

   flow = Flow.from_records(gen())

.. warning::
   If you mutate records (e.g. with ``.assign()``), Flow modifies them in place. Use ``.copy()`` or ``deepcopy()`` to protect your originals.

📄 From JSON Lines (JSONL) File: ``.from_jsonl(...)``
=====================================================

.. code-block:: python

   flow = Flow.from_jsonl("data/events.jsonl")

📂 From Folder of JSON Files: ``.from_folder(...)``
===================================================

.. code-block:: python

   flow = Flow.from_folder("data/events/")

Reads all ``.json`` and ``.jsonl`` files in a directory.

Each ``.json`` file must contain either:

- A single dict
- A list of dicts
- Files are streamed one at a time - efficient for bulk ingestion.

✨ From Glob Pattern: ``.from_glob(...)``
=========================================

.. code-block:: python

   flow = Flow.from_glob("data/**/*.json")

Searches recursively using ``glob.glob``. Same behavior as ``.from_folder``, but more flexible for matching paths and subfolders.

🧾 From JSON File (Single Object or Array): ``.from_json(...)``
===============================================================

.. code-block:: python

   flow = Flow.from_json("data/game.json")

- Accepts a single object (as one record), or
- A list of objects (as multiple records)

.. note::
   This reads the entire file into memory. Use ``.from_jsonl()`` for streaming large datasets.

Working with Cloud Storage (S3, GCS, Azure)
============================================

All file-based creation methods (``from_json``, ``from_jsonl``, ``from_folder``, ``from_glob``) can read directly from cloud storage by providing the appropriate URI and storage_options.

To do this, you'll need to install the necessary dependencies for your cloud provider:

- **Amazon S3**: pip install penaltyblog[aws]
- **Google Cloud Storage**: pip install penaltyblog[gcp]
- **Azure Data Lake / Blob Storage**: pip install penaltyblog[azure]

The ``storage_options`` parameter is an optional dictionary containing your credentials if you are not storing them as environment variables.

.. code-block:: python

   import penaltyblog as pb

   s3_options = {
       "key": "YOUR_AWS_ACCESS_KEY_ID",
       "secret": "YOUR_AWS_SECRET_ACCESS_KEY",
   }
   flow = pb.Flow.from_json("s3://my-bucket/data.json", storage_options=s3_options)

   gcs_options = {"token": "path/to/your/gcs_credentials.json"}
   flow = pb.Flow.from_jsonl("gs://my-gcs-bucket/data.jsonl", storage_options=gcs_options)

   azure_options = {
       "account_name": "YOUR_STORAGE_ACCOUNT_NAME",
       "account_key": "YOUR_STORAGE_ACCOUNT_KEY",
   }
   flow = pb.Flow.from_folder("abfs://container/data/", storage_options=azure_options)

💾 Saving Data from a Flow
==========================

Once your pipeline is complete, use ``.to_*()`` methods to export the result.

``.to_jsonl(path)``
-------------------

Write one record per line:

.. code-block:: python

   flow.to_jsonl("output/events.jsonl")

``.to_json(path)``
------------------

Write all records as a JSON array:

.. code-block:: python

   flow.to_json("summary.json", indent=4)

.. note::
   This collects the entire stream before writing.

``.to_json_files(folder, by="id")``
-----------------------------------

Write each record to its own .json file:

.. code-block:: python

   flow.to_json_files("out/", by="event_id")

- "out/123.json"
- "out/456.json"

Field must be a string or something serializable to filename.

``.to_pandas()``
----------------

Convert the flow to a Pandas DataFrame:

.. code-block:: python

   df = flow.select("player_name", "shot_xg").to_pandas()

.. note::
   Best used after filtering/flattening to avoid deeply nested fields.

✅ Summary
==========

Input Options
-------------

+------------------+-------------------------+------------+------------------------------+
| Source Format    | Method                  | Streaming? | Notes                        |
+==================+=========================+============+==============================+
| Python objects   | ``.from_records()``     | ✅         | Lists, dicts, or generators  |
+------------------+-------------------------+------------+------------------------------+
| JSONL file       | ``.from_jsonl()``       | ✅         | Efficient for large datasets |
+------------------+-------------------------+------------+------------------------------+
| Single JSON file | ``.from_json()``        | ❌         | Loads entire file at once    |
+------------------+-------------------------+------------+------------------------------+
| Folder of files  | ``.from_folder()``      | ✅         | Streams one file at a time   |
+------------------+-------------------------+------------+------------------------------+
| Glob pattern     | ``.from_glob()``        | ✅         | Recursively matches files    |
+------------------+-------------------------+------------+------------------------------+

Output Options
--------------

+--------------------+-----------------+------------+-------------------------------+
| Output Method      | Format          | Streaming? | Notes                         |
+====================+=================+============+===============================+
| ``.to_jsonl()``    | JSONL           | ✅         | One line per record           |
+--------------------+-----------------+------------+-------------------------------+
| ``.to_json()``     | JSON array      | ❌         | Collects before writing       |
+--------------------+-----------------+------------+-------------------------------+
| ``.to_json_files()`` | Folder of files | ✅         | One file per record           |
+--------------------+-----------------+------------+-------------------------------+
| ``.to_pandas()``   | DataFrame       | ❌         | Collects all data into memory |
+--------------------+-----------------+------------+-------------------------------+

🧠 What's Next?
===============

Now that you can load and save data, let's look at inspecting, debugging, and explaining your flows using ``.head()``, ``.keys()``, ``.explain()`` and more.
