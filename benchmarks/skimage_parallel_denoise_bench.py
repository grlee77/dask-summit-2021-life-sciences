import numpy as np
from skimage import data, img_as_float
from skimage.util import apply_parallel
from skimage import restoration

img = img_as_float(data.camera())
img = np.tile(img, (4, 8))
sigma = 0.15
rstate = np.random.RandomState(1234)
img = img + sigma * rstate.standard_normal(img.shape)

chunks = data.camera().shape

sigma_est = restoration.estimate_sigma(img)


for func, kwargs in [
    (restoration.denoise_wavelet, dict(sigma=sigma_est, multichannel=False, wavelet='sym4', wavelet_levels=3))
    (restoration.denoise_tv_chambolle, )]


extra_kwargs = dict(sigma=sigma_est, multichannel=False, wavelet='sym4', wavelet_levels=3)
dn_wav = apply_parallel(restoration.denoise_wavelet, img, chunks=chunks, depth=8, extra_keywords=extra_kwargs, dtype=img.dtype, multichannel=False, compute=True)
restoration.denoise_wavelet(img, sigma=sigma_est, multichannel=False, wavelet='sym4', wavelet_levels=3)
# 453 ms with apply_parallel vs. 783 ms without

extra_keywords = dict(weight=0.1, n_iter_max=100, multichannel=False)
dn_tv_c = apply_parallel(restoration.denoise_tv_chambolle, img, chunks=chunks, depth=16, extra_keywords=extra_keywords, dtype=img.dtype, multichannel=False, compute=True)
restoration.denoise_tv_chambolle(img, **extra_keywords)
# 3.98 s ± 41.6 ms with apply_parallel vs. 12.1 s +/- 58.5 ms without

extra_keywords = dict(weight=1, max_iter=100)  # , multichannel=False)
%timeit dn_tv_b = apply_parallel(restoration.denoise_tv_bregman, img, chunks=chunks, depth=16, extra_keywords=extra_keywords, dtype=img.dtype, compute=True)
%timeit restoration.denoise_tv_bregman(img, **extra_keywords)
# # 3.2 s ± 110 ms with apply_parallel vs. 2.35 s ± 27.9 ms without
# 529 ms with apply_parallel vs. 2.33 s +/- 57.1 ms without

patch_distance = 5
patch_size = 5
depth = patch_distance + patch_size // 2
extra_keywords = dict(patch_size=5, patch_distance=5, h=0.4*sigma, sigma=sigma, multichannel=False, fast_mode=True, preserve_range=False)
%timeit dn_nlm = apply_parallel(restoration.denoise_nl_means, img, chunks=chunks, depth=depth, extra_keywords=extra_keywords, dtype=img.dtype, multichannel=False, compute=True)
%timeit restoration.denoise_nl_means(img, **extra_keywords)
# 1.25 s ± 24.7 ms with apply_parallel vs. 8.37 s ± 124 ms without


img_clip = np.clip(img, 0, 1)
extra_keywords = {}
%timeit dn_bilateral = apply_parallel(restoration.denoise_bilateral, img_clip, chunks=chunks, depth=8, extra_keywords=extra_keywords, dtype=img.dtype, compute=True)
%timeit restoration.denoise_bilateral(img_clip, **extra_keywords)
# -> 5.23 s ± 152 ms  with apply_parallel vs. 3.79 s ± 41.8 ms without
# 811 ms denoise_bilateral with apply_parallel vs. 3.75 s +/- 87.1 ms without

from skimage.filters import rank
import dask

selem = np.ones((5, 5), dtype=np.uint8)
depth = selem.shape[0] // 2
img_uint = (np.clip(img, 0, 1) * 255).astype(np.uint8)
extra_keywords = dict(selem=selem)
#with dask.config.set(scheduler='threads'):  # scheduler='processes'):
%timeit rank_median = apply_parallel(rank.median, img_uint, chunks=chunks, depth=depth, extra_keywords=extra_keywords, dtype=img.dtype, compute=True)
%timeit rank.median(img_uint, **extra_keywords)
#   vs. 1.48s +/- 28.2 ms without