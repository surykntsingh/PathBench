#!/usr/bin/env python3
"""Convert every LMDB in a directory to HDF5 files in the same directory."""

import argparse
import io
import pickle
import shutil
from pathlib import Path

import h5py
import lmdb
import numpy as np
from tqdm import tqdm


METADATA_KEYS = {b"__keys__", b"__len__", b"length", b"keys"}


def parse_args():
    parser = argparse.ArgumentParser(description="Replace LMDB files/directories with .h5 files.")
    parser.add_argument("directory", type=Path, help="Directory containing LMDB files or directories.")
    return parser.parse_args()


def is_lmdb(path):
    return (path.is_file() and path.suffix == ".lmdb") or (
        path.is_dir() and (path / "data.mdb").exists()
    )


def output_path_for(lmdb_path):
    if lmdb_path.suffix:
        return lmdb_path.with_suffix(".h5")
    return lmdb_path.with_name(f"{lmdb_path.name}.h5")


def read_payload(value):
    try:
        payload = pickle.loads(value)
    except Exception:
        with io.BytesIO(value) as buffer:
            payload = np.load(buffer, allow_pickle=False)

    if hasattr(payload, "detach"):
        payload = payload.detach().cpu().numpy()

    return np.asarray(payload)


def read_lmdb(lmdb_path):
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=lmdb_path.is_dir(),
    )

    try:
        with env.begin(write=False) as txn:
            keys_blob = txn.get(b"__keys__")
            if keys_blob is not None:
                keys = pickle.loads(keys_blob)
                keys = [key if isinstance(key, bytes) else str(key).encode("utf-8") for key in keys]
            else:
                keys = [key for key, _ in txn.cursor() if key not in METADATA_KEYS]

            if len(keys) != 1:
                raise ValueError(f"expected exactly one data record, found {len(keys)}")

            value = txn.get(keys[0])
            if value is None:
                raise ValueError(f"missing value for key {keys[0]!r}")

            return read_payload(value)
    finally:
        env.close()


def remove_lmdb(lmdb_path):
    if lmdb_path.is_dir():
        shutil.rmtree(lmdb_path)
    else:
        lmdb_path.unlink()


def convert_lmdb(lmdb_path):
    h5_path = output_path_for(lmdb_path)
    features = read_lmdb(lmdb_path)

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("features", data=features)

    remove_lmdb(lmdb_path)
    return h5_path


def main():
    directory = parse_args().directory
    lmdb_paths = sorted(path for path in directory.iterdir() if is_lmdb(path))

    for lmdb_path in tqdm(lmdb_paths, desc="Converting LMDBs", unit="lmdb"):
        h5_path = convert_lmdb(lmdb_path)
        tqdm.write(f"{lmdb_path.name} -> {h5_path.name}")


if __name__ == "__main__":
    main()
