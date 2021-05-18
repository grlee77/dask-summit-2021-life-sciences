import os
import pickle

import cupy
import cupy as cp
import cupyx.scipy.ndimage
import dask.array as da
import dask_image
import dask_image.ndfilters
import numpy as np
import pandas as pd
import scipy

from _image_bench import ImageBench


class ConvolveBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        weights_shape,
        dtypes=[np.float32],
        fixed_kwargs={},
        var_kwargs={},
        chunks='auto',
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        module_dask=dask_image.ndfilters,
    ):

        self.weights_shape = weights_shape

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            module_dask=module_dask,
            chunks=chunks,
        )

    def set_args(self, dtype):
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype)
        image = cp.asnumpy(imaged)

        wd = cupy.testing.shaped_random(self.weights_shape, xp=cp, dtype=dtype)
        w = cp.asnumpy(wd)

        self.args_cpu = (image, w)
        self.args_dask_cpu = (da.from_array(image, chunks=self.chunks), w)
        self.args_gpu = (imaged, wd)


class FilterBench(ImageBench):
    def set_args(self, dtype):
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_dask_cpu = (da.from_array(image, chunks=self.chunks),)
        self.args_gpu = (imaged,)


pfile = "filter_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

modes = ["constant", "mirror"]
prefilter = True
dtypes = [np.float32]
for shape in [(3840, 2160), (7680, 4320), (192, 192, 192), (512, 256, 256)]:
    ndim = len(shape)
    weights_shape = (5,) * ndim
    weights_shape1d = weights_shape[:1]

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

    for fname, var_kwargs in [
        # ("uniform_filter", dict(mode=["nearest"], size=[3, 7, 11])),
        # ("gaussian_filter", dict(mode=["nearest"], sigma=[0.33, 1, 3, 9])),
        # ("maximum_filter", dict(mode=["nearest"], size=[3, 5, 7])),
        # ("minimum_filter", dict(mode=["nearest"], size=[3, 5, 7])),
        ("median_filter", dict(mode=["nearest"], size=[3, 5, 7, 11])),
        # ("percentile_filter", dict(mode=["nearest"], size=[3, 5, 7], percentile=[30])),
        # ("rank_filter", dict(mode=["nearest"], size=[3, 5, 7], rank=[-2])),
        # ("prewitt", dict(mode=["nearest"], axis=[0, -1])),
        # ("sobel", dict(mode=["nearest"], axis=[0, -1])),
        # ("laplace", dict(mode=["nearest"])),
        # ("gaussian_laplace", dict(mode=["nearest"], sigma=[0.33, 3, 9])),
        # ("gaussian_gradient_magnitude", dict(mode=["nearest"], sigma=[0.33, 3, 9])),

        # dask_image doesn't have these 1d variants
        # (
        #     "gaussian_filter1d",
        #     dict(mode=["nearest"], sigma=[0.33, 3, 9], axis=[0, -1], order=[0, 1]),
        # ),
        # ("uniform_filter1d", dict(mode=["nearest"], size=[3, 7, 11], axis=[0, -1])),
        # ("maximum_filter1d", dict(mode=["nearest"], size=[3, 7], axis=[0, -1])),
        # ("minimum_filter1d", dict(mode=["nearest"], size=[3, 7], axis=[0, -1])),

    ]:
        for chunks in all_chunks:
        # TODO: add cases for generic_filter and generic_filter1d?


            if ndim == 3 and fname == "median_filter":
                var_kwargs["size"] = [3, 5, 7]  # omit sizes > 5

            B = FilterBench(
                function_name=fname,
                shape=shape,
                dtypes=dtypes,
                #fixed_kwargs=dict(output=None),
                chunks=chunks,
                var_kwargs=var_kwargs,
                module_dask=dask_image.ndfilters,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])

    for fname, wshape, var_kwargs in [
#        ("convolve", weights_shape, dict(mode=modes)),
#        ("correlate", weights_shape, dict(mode=modes)),
        # dask_image doesn't have these 1D variants
        # ("convolve1d", weights_shape1d, dict(mode=modes, axis=[0, -1])),
        # ("correlate1d", weights_shape1d, dict(mode=modes, axis=[0, -1])),
    ]:
        for chunks in all_chunks:
        # TODO: add cases for generic_filter and generic_filter1d?

            B = ConvolveBench(
                function_name=fname,
                shape=shape,
                weights_shape=wshape,
                dtypes=dtypes,
                #fixed_kwargs=dict(output=None, origin=0),
                fixed_kwargs=dict(origin=0),
                chunks=chunks,
                var_kwargs=var_kwargs,
                module_dask=dask_image.ndfilters,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
