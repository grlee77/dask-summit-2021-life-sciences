# Dask Summit 2021 - image processing benchmarks

This respoistory contains benchmarking and plotting scripts used to generate
performance comparisons for dask-image, SciPy and CuPy for a lightning talk at
Dask Summit 2021. The functions tested were restricted to those present in
all three libraries.

## Timing scikit-image restoration functions accelerated with apply_parallel

There is an example notebook for this located at
`benchmarks/skimage_apply_parallel.ipynb`

Recent versions of dask and scikit-image will be required. It has only been
tested with

scikit-image >=0.18
dask >= 2020.12.0

A recent PR for scikit-image fixed multi-threaded use of `denoise_tv_bregman`
and `denoise_bilateral`, but has not yet appeared in a released version of the
library:

https://github.com/scikit-image/scikit-image/pull/5400


## Dask/CuPy/SciPy benchmarks

The benchmarking scripts named `dask_cupyx_scipy_*` were adapted from ones
[previously created for the cuCIM library](https://github.com/rapidsai/cucim/tree/branch-21.06/benchmarks/skimage).

These scripts generate Markdown tables and CSV format outputs containing
benchmark results.

Selected results were manually copied into plotting scripts in
`benchmarks/viz`.

### Requirements for Dask/CuPy/SciPy benchmarks

CuPy >= 9
SciPy >= 1.6
pandas
dask-image (pre-release)

Running the benchmarks currently requires a branch incorporating a few recent
PRs for dask-image that were opened after the release of v0.6.0:

https://github.com/dask/dask-image/pull/215

https://github.com/dask/dask-image/pull/221

https://github.com/dask/dask-image/pull/222

These may be included in the next release of dask-image.
