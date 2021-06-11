# adapted from cuCIM, see ../LICENSE-3rdparty.txt

import math
import os
import pickle
import skimage
import skimage.restoration

# import dask.array as da
# import dask_image
# import dask_image.ndmorph
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as ndi

from _image_bench_apply_parallel import ApplyParallelBench

_sigma = 0.05

class DenoiseBench(ApplyParallelBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]

        # add noise
        if np.dtype(dtype).kind in "iu":
            sigma = _sigma * 255
            im1 = im1 + sigma * np.random.randn(*im1.shape)
            im1 = np.clip(im1, 0, 255).astype(dtype)
        else:
            sigma = _sigma
            im1 = im1 + sigma * np.random.randn(*im1.shape)
            im1 = np.clip(im1, 0, None)  # non-negative for denoise_bilateral

        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        image = np.tile(im1, n_tile)[slices]

        self.args_cpu = (image,)


class DeconvolutionBench(ApplyParallelBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]
        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        image = np.tile(im1, n_tile)[slices]

        psf = np.ones((5,) * image.ndim) / 25
        image = ndi.convolve(image, psf)

        self.args_cpu = (image, psf)


pfile = "apply_parallel_restoration.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

dtypes = [np.float32]

for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd, depth in [
    # _denoise.py
    ("denoise_tv_chambolle", dict(), dict(weight=[0.02]), True, True, 4),
    ("denoise_tv_bregman", dict(max_iter=25), dict(weight=[0.02]), True, True, 4),
    ("denoise_bilateral", dict(), dict(), True, True, 4),
    ("denoise_wavelet", dict(wavelet='sym2', wavelet_levels=3), dict(), True, False, 8),
    ("denoise_nl_means", dict(fast_mode=True, sigma=_sigma, h=0.6*_sigma), dict(), True, True, 8),
    # # j_invariant.py
    # ("calibrate_denoiser", dict(), dict(), False, True),
]:

    for shape in [(3840, 2160), (7680, 4320), (192, 192, 192)]: # , (512, 256, 256)]:

        ndim = len(shape)
        if not allow_nd:
            if not allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue
        if shape[-1] == 3 and not allow_color:
            continue

        if ndim == 2:
            all_chunks = [
                (shape[0] // 5, shape[1] // 4),
            ]
        else:
            all_chunks = [
                (shape[0] // 4, shape[1] // 2, shape[2] // 2),
            ]

        if function_name == "denoise_nl_means":
            fixed_kwargs["multichannel"] = shape[-1] == 3
            fixed_kwargs["patch_size"] = 5 if ndim == 2 else 3
            fixed_kwargs["patch_distance"] = 5 if ndim == 2 else 2
            depth = fixed_kwargs["patch_distance"] + fixed_kwargs["patch_size"] // 2

        if function_name == "denoise_tv_chambolle":
            fixed_kwargs["multichannel"] = shape[-1] == 3

        if function_name == "calibrate_denoiser":
            denoise_class = CalibratedDenoiseBench
        else:
            denoise_class = DenoiseBench

        for chunks in all_chunks:
            B = denoise_class(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.restoration,
                apply_parallel_kwargs=dict(chunks=chunks, depth=depth)
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])


# function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd = ('unsupervised_wiener', dict(), dict(), False, True)
dtype = np.float32
for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd, depth in [
    # deconvolution.py
    ("wiener", dict(balance=100.0), dict(), False, False, 0),
    ("unsupervised_wiener", dict(), dict(), False, False, 0),
    ("richardson_lucy", dict(), dict(iterations=[5]), False, True, 8),
]:

    for shape in [(3840, 2160), (7680, 4320), (192, 192, 192), (512, 256, 256)]:

        ndim = len(shape)
        if not allow_nd:
            if not allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue
        if shape[-1] == 3 and not allow_color:
            continue

        if ndim == 2:
            all_chunks = [
                (shape[0] // 5, shape[1] // 4),
            ]
        else:
            all_chunks = [
                (shape[0] // 4, shape[1] // 2, shape[2] // 2),
            ]

        output_wrapper = 'first_output_only' \
                          if function_name == 'unsupervised_wiener' \
                          else None
        for chunks in all_chunks:

            B = DeconvolutionBench(
                function_name=function_name,
                shape=shape,
                dtypes=[dtype],
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.restoration,
                apply_parallel_kwargs=dict(chunks=chunks, depth=depth, dtype=dtype),
                output_wrapper=output_wrapper,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
