import math
import os
import pickle

import cupy
import cupy as cp
import cupyx.scipy.ndimage
import dask.array as da
import dask_image
import dask_image.ndmeasure
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as ndi

from _image_bench import ImageBench


class LabelBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        structure=None,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        chunks='auto',
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        module_dask=dask_image.ndmeasure,
    ):

        self.contiguous_labels = contiguous_labels
        array_kwargs = dict(structure=structure)
        if "structure" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'structure'")
        fixed_kwargs.update(array_kwargs)

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            chunks=chunks,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            index_str=index_str,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            module_dask=module_dask,
        )

    def set_args(self, dtype):
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ]
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        image_dask = da.from_array(image, chunks=self.chunks)
        self.args_cpu = (image,)
        self.args_dask_cpu = (image_dask,)
        self.args_gpu = (imaged,)


class MeasurementsBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        use_index=False,
        nlabels=16,
        dtypes=[np.float32],
        chunks='auto',
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        module_dask=dask_image.ndmeasure,
    ):

        self.nlabels = nlabels
        self.use_index = use_index
        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            chunks=chunks,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            index_str=index_str,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            module_dask=module_dask,
        )

    def set_args(self, dtype):
        size = math.prod(self.shape)
        imaged = cupy.arange(size, dtype=dtype).reshape(self.shape)
        labelsd = cupy.random.choice(self.nlabels, size)
        labelsd = labelsd.reshape(self.shape) + 1

        image = cp.asnumpy(imaged)
        labels = cp.asnumpy(labelsd)
        if self.use_index:
            indexd = cupy.arange(1, self.nlabels + 1, dtype=cupy.intp)
            index = cp.asnumpy(indexd)
            # index_dask = da.from_array(index, chunks=self.chunks)
        else:
            indexd = None
            index = None
            index_dask = None

        image_dask = da.from_array(image, chunks=self.chunks)
        labels_dask = da.from_array(labels, chunks=self.chunks)

        self.args_cpu = (image,)
        self.args_gpu = (imaged,)
        self.args_dask_cpu = (image_dask,)

        # store labels and index as fixed_kwargs since histogram does not use
        # them in the same position
        self.fixed_kwargs_gpu.update(dict(labels=labelsd, index=indexd))
        self.fixed_kwargs_dask_cpu.update(
            dict(label_image=labels_dask, index=index)  # Note: dask_image uses the name "label_image" instead of "labels"
        )
        self.fixed_kwargs_cpu.update(dict(labels=labels, index=index))


pfile = "measurements_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

dtypes = [np.float32]
for shape in [(3840, 2160), (7680, 4320), ]: # (192, 192, 192), (512, 256, 256)]:
    ndim = len(shape)

    if ndim == 2:
        all_chunks = [
            # (shape[0] // 5, shape[1] // 2),
            (shape[0] // 5, shape[1] // 4),
            # (shape[0] // 20, shape[1] // 1),
            # (shape[0] // 20, shape[1] // 2),
        ]
    else:
        all_chunks = [
            (shape[0] // 4, shape[1] // 2, shape[2] // 2),
            # (shape[0] // 5, shape[1] // 2, shape[2] // 2),
            # (shape[0] // 8, shape[1] // 4, shape[2])
        ]

    if ndim == 3:
        raise ValueError("3D test data not implemented")

    for fname, var_kwargs in [
        ("label", {}),  # dict(greyscale_mode=[False, True]) not available in cupyx
    ]:
        for contiguous_labels in [True, False]:
            if contiguous_labels:
                index_str = "contiguous"
            else:
                index_str = None

            for chunks in all_chunks:
                B = LabelBench(
                    function_name=fname,
                    shape=shape,
                    dtypes=dtypes,
                    chunks=chunks,
                    structure=ndi.generate_binary_structure(ndim, ndim),
                    contiguous_labels=contiguous_labels,
                    index_str=index_str,
                    fixed_kwargs={},  # dict(output=None),
                    var_kwargs=var_kwargs,
                    module_dask=dask_image.ndmeasure,
                )
                results = B.run_benchmark(duration=1)
                all_results = all_results.append(results["full"])

    for fname in [
        "sum",
        # "mean",
        # "variance",
        # "standard_deviation",
        # "minimum",
        # "minimum_position",
        # "maximum",
        # "maximum_position",
        # "median",
        # "extrema",
        # "center_of_mass",
    ]:
        for use_index in [True, False]:
            if use_index:
                nlabels_cases = [4, 16, 64, 256]
            else:
                nlabels_cases = [16]

            for nlabels in nlabels_cases:
                if use_index:
                    index_str = f"{nlabels} labels, no index"
                else:
                    index_str = f"{nlabels} labels, with index"


                for chunks in all_chunks:
                    B = MeasurementsBench(
                        function_name=fname,
                        shape=shape,
                        dtypes=dtypes,
                        chunks=chunks,
                        use_index=use_index,
                        nlabels=nlabels,
                        index_str=index_str,
                        var_kwargs=var_kwargs,
                        module_dask=dask_image.ndmeasure,
                    )
                    results = B.run_benchmark(duration=1)
                    all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
