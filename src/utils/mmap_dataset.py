import os
import shutil
import struct
from functools import lru_cache

import numpy as np
import torch

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
}


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class DatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndex(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        def write_index(path, dtype, sizes):
            def _get_pointers(sizes):
                dtype_size = dtype().itemsize
                address = 0
                pointers = []

                for size in sizes:
                    pointers.append(address)
                    address += size * dtype_size

                return pointers

            with open(path, "wb") as _file:
                _file.write(struct.pack("<B", code(dtype)))
                pointers = _get_pointers(sizes)

                _file.write(struct.pack("<Q", len(sizes)))

                sizes = np.array(sizes, dtype=np.int32)
                _file.write(sizes.tobytes(order="C"))
                del sizes

                pointers = np.array(pointers, dtype=np.int64)
                _file.write(pointers.tobytes(order="C"))
                del pointers

        write_index(index_file, self._dtype, self._sizes)


class MMapIndex(object):
    def __init__(self, path):
        with open(path, "rb") as stream:
            (dtype_code,) = struct.unpack("<B", stream.read(1))
            self._dtype = dtypes[dtype_code]
            self._dtype_size = self._dtype().itemsize

            self._len = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        _warmup_mmap_file(path)

        self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        self._sizes = np.frombuffer(
            self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
        )
        self._pointers = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._len,
            offset=offset + self._sizes.nbytes,
        )

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    @property
    def dtype(self):
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self._pointers[i], self._sizes[i]

    def __len__(self):
        return self._len


class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = MMapIndex(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
