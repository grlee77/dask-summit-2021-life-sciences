import math
import os
import pickle

import cupy
import cupy as cp
import cupyx.scipy.ndimage
import dask.array as da
import dask_image
import dask_image.ndmorph
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as ndi

from _image_bench import ImageBench


class BinaryMorphologyBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        structure=None,
        # mask=None,
        dtypes=[np.float32],
        fixed_kwargs={},
        var_kwargs={},
        index_str="",
        chunks="auto",
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        module_dask=dask_image.ndmorph,
    ):

        array_kwargs = dict(structure=structure)  # , mask=mask)
        if "structure" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'structure'")
        if "mask" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'mask'")
        fixed_kwargs.update(array_kwargs)

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            index_str=index_str,
            chunks=chunks,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            module_dask=module_dask,
        )

    def set_args(self, dtype):
        imaged = cp.random.standard_normal(self.shape).astype(dtype) > 0
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_dask_cpu = (da.from_array(image, chunks=self.chunks),)
        self.args_gpu = (imaged,)


class MorphologyBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        structure=None,
        footprint=None,
        dtypes=[np.float32],
        fixed_kwargs={},
        var_kwargs={},
        chunks='auto',
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        module_dask=dask_image.ndmorph,
    ):

        array_kwargs = dict(structure=structure, footprint=footprint)
        if "structure" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'structure'")
        if "footprint" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'footprint'")
        fixed_kwargs.update(array_kwargs)

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            chunks=chunks,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            module_dask=module_dask,
        )

    def set_args(self, dtype):
        imaged = cp.random.standard_normal(self.shape).astype(dtype)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_dask_cpu = (da.from_array(image, chunks=self.chunks),)
        self.args_gpu = (imaged,)


pfile = "morphology_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

modes = ["reflect"]
sizes = [3, 5, 7, 9]

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

    # for fname, var_kwargs in [
    #     ("grey_erosion", dict(mode=modes, size=sizes)),
    #     ("grey_dilation", dict(mode=modes, size=sizes)),
    #     ("grey_opening", dict(mode=modes, size=sizes)),
    #     ("grey_closing", dict(mode=modes, size=sizes)),
    #     ("morphological_gradient", dict(mode=modes, size=sizes)),
    #     ("morphological_laplace", dict(mode=modes, size=sizes)),
    #     ("white_tophat", dict(mode=modes, size=sizes)),
    #     ("black_tophat", dict(mode=modes, size=sizes)),
    # ]:
    #     for chunks in all_chunks:
    #         B = MorphologyBench(
    #             function_name=fname,
    #             shape=shape,
    #             dtypes=dtypes,
    #             chunks=chunks,
    #             structure=None,
    #             footprint=None,
    #             # Note: Benchmark runner will change brute_force to True for the GPU
    #             fixed_kwargs={},  # dict(output=None),
    #             var_kwargs=var_kwargs,
    #             module_dask=dask_image.ndmorph,
    #         )
    #         results = B.run_benchmark(duration=1)
    #         all_results = all_results.append(results["full"])

    iterations = [1, 5]
    for fname, var_kwargs in [
        ("binary_erosion", dict(iterations=iterations, brute_force=[False])),
        ("binary_dilation", dict(iterations=iterations, brute_force=[False])),
        ("binary_opening", dict(iterations=iterations, brute_force=[False])),
        ("binary_closing", dict(iterations=iterations, brute_force=[False])),
        # ("binary_propagation", dict()),
    ]:
        # for connectivity in (list(range(1, ndim + 1)) + [sq7]):
        #     for chunks in all_chunks:
        #         index_str = f"conn={connectivity}"
        #         structure = ndi.generate_binary_structure(ndim, connectivity)

        #         B = BinaryMorphologyBench(
        #             function_name=fname,
        #             shape=shape,
        #             dtypes=dtypes,
        #             structure=structure,
        #             # mask=None,
        #             chunks=chunks,
        #             index_str=index_str,
        #             # Note: Benchmark runner will change brute_force to True for the GPU
        #             fixed_kwargs={}, # dict(output=None),
        #             var_kwargs=var_kwargs,
        #             module_dask=dask_image.ndmorph,
        #         )
        #         results = B.run_benchmark(duration=1)
        #         all_results = all_results.append(results["full"])

        for side_length in [5, 7]:
            structure = np.ones((side_length,) * ndim, dtype=np.uint8)
            for chunks in all_chunks:
                index_str = f"square{side_length}"

                B = BinaryMorphologyBench(
                    function_name=fname,
                    shape=shape,
                    dtypes=dtypes,
                    structure=structure,
                    # mask=None,
                    chunks=chunks,
                    index_str=index_str,
                    # Note: Benchmark runner will change brute_force to True for the GPU
                    fixed_kwargs={}, # dict(output=None),
                    var_kwargs=var_kwargs,
                    module_dask=dask_image.ndmorph,
                )
                results = B.run_benchmark(duration=1)
                all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
