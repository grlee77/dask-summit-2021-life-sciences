# benchmarking the n-dimensional image API using dask-image, SciPy and CuPy

This respoistory contains benchmarking and plotting scripts used to generate performance comparisons for dask-image, SciPy and CuPy for a lightning talk at Dask Summit 2021. The functions tested were restricted to those present across all three libraries.

The benchmarking scripts were adapted from ones previously created for the cuCIM library:
https://github.com/rapidsai/cucim/tree/branch-21.06/benchmarks/skimage

Running the benchmarks currently requires a branch incorporating a few recent PRs for dask-image that were opened after the release of v0.6.0:
https://github.com/dask/dask-image/pull/215
https://github.com/dask/dask-image/pull/221
https://github.com/dask/dask-image/pull/222
Hopefully these will all be included in the next release of dask-image.

Similarly, the following recent PR for scikit-image fixed multi-threaded use of denoise_tv_bregman and denoise_bilateral, but has not yet appeared in a released version of the library:
https://github.com/scikit-image/scikit-image/pull/5400
