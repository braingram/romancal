import tempfile
from pathlib import Path

import numpy as np


# not inheriting from MutableSequence here as insert is complicated
class ArrayLibrary:
    def __init__(self, tempdir=""):
        self._temp_dir = tempfile.TemporaryDirectory(dir=tempdir)
        self._temp_path = Path(self._temp_dir.name)
        self._filenames = []
        self._data_shape = None
        self._data_dtype = None

    @property
    def closed(self):
        return not hasattr(self, "_temp_dir")

    def close(self):
        if self.closed:
            return
        self._temp_dir.cleanup()
        del self._temp_dir

    def __del__(self):
        self.close()

    def __len__(self):
        if self.closed:
            raise Exception("use after close")
        return len(self._filenames)

    def __getitem__(self, index):
        if self.closed:
            raise Exception("use after close")
        fn = self._filenames[index]
        return np.load(fn)

    def _validate_input(self, arr):
        if arr.ndim != 2:
            raise Exception(f"Only 2D arrays are supported: {arr.ndim}")
        if self._data_shape is None:
            self._data_shape = arr.shape
        else:
            if arr.shape != self._data_shape:
                raise Exception(
                    f"Input shape mismatch: {arr.shape} != {self._data_shape}"
                )
        if self._data_dtype is None:
            self._data_dtype = arr.dtype
        else:
            if arr.dtype != self._data_dtype:
                raise Exception(
                    f"Input dtype mismatch: {arr.dtype} != {self._data_dtype}"
                )

    def __setitem__(self, index, value):
        self._validate_input(value)
        if self.closed:
            raise Exception("use after close")
        fn = self._filenames[index]
        if fn is None:
            fn = self._temp_path / f"{index}.npy"
        np.save(fn, value, False)
        self._filenames[index] = fn

    def append(self, value):
        if self.closed:
            raise Exception("use after close")
        index = len(self)
        self._filenames.append(None)
        self.__setitem__(index, value)

    def median(self, buffer_size=100 << 20):
        if self.closed:
            raise Exception("use after close")
        if not len(self):
            raise Exception("can't take median of empty list")

        # figure out how big the buffer can be
        n_arrays = len(self)
        allowed_memory_per_array = buffer_size // n_arrays

        n_dim_1 = allowed_memory_per_array // (
            self._data_dtype.itemsize * self._data_shape[0]
        )
        if n_dim_1 < 1:
            # TODO more useful error message
            raise Exception("Not enough memory")
        if n_dim_1 >= self._data_shape[1]:
            return np.nanmedian(self, axis=0)

        buffer = np.empty(
            (n_arrays, self._data_shape[0], n_dim_1), dtype=self._data_dtype
        )
        median = np.empty(self._data_shape, dtype=self._data_dtype)

        e = n_dim_1
        slices = [slice(0, e)]
        while e <= self._data_shape[1]:
            s = e
            e += n_dim_1
            slices.append(slice(s, min(e, self._data_shape[1])))

        for s in slices:
            for i, arr in enumerate(self):
                # TODO is it more efficient to slice on a different axis?
                buffer[i, :, : (s.stop - s.start)] = arr[:, s]
            median[:, s] = np.nanmedian(buffer[:, :, : (s.stop - s.start)], axis=0)
        return median
