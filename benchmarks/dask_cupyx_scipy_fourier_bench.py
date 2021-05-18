import os
import pickle

import cupy
import cupy as cp
import numpy as np
import dask.array as da
import dask_image
import dask_image.ndfourier
import pandas as pd

from _image_bench import ImageBench

pfile = "fourier_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

dtypes = [np.float32]

for shape in [(3840, 2160), (7680, 4320), (192, 192, 192), (512, 256, 256)]:
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

    class FourierBench(ImageBench):
        def set_args(self, dtype):
            cplx_dt = np.promote_types(dtype, np.complex64)
            imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=cplx_dt)
            image = cp.asnumpy(imaged)
            self.args_cpu = (image,)
            self.args_dask_cpu = (da.from_array(image, chunks=self.chunks),)
            self.args_gpu = (imaged,)

    for fname, fixed_kwargs, var_kwargs in [
        ("fourier_gaussian", dict(sigma=5), {}),
        ("fourier_uniform", dict(size=16), {}),
        ("fourier_shift", dict(shift=5), {}),
        # ellipsoid not implemented in Dask case
        # ("fourier_ellipsoid", dict(size=15.0), {}),
    ]:
        for chunks in all_chunks:

            B = FourierBench(
                function_name=fname,
                shape=shape,
                dtypes=dtypes,
                chunks=chunks,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_dask=dask_image.ndfourier,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])


fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
