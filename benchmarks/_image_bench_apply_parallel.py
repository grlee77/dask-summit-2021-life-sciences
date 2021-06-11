import copy
import itertools
import math
import time
import types
from collections import abc
import re
import subprocess

# import dask.array as da
# import dask_image
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.data
from skimage.util import apply_parallel

from _time_cpu import repeat


def product_dict(**kwargs):
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class ApplyParallelBench(object):
    def __init__(
        self,
        function_name,
        shape,
        dtypes=[np.float32],
        fixed_kwargs={},
        var_kwargs={},
        chunks='auto',
        index_str=None,  # extra string to append to dataframe index
        # set_args_kwargs={},  # for passing additional arguments to custom set_args method
        module_cpu=scipy.ndimage,
        function_is_generator=False,
        apply_parallel_kwargs={},
        output_wrapper=None
    ):

        self.shape = shape
        self.function_name = function_name
        self.fixed_kwargs_cpu = fixed_kwargs
        self.var_kwargs = var_kwargs
        self.index_str = index_str
        self.chunks = chunks
        # self.set_args_kwargs = set_args_kwargs
        if not isinstance(dtypes, abc.Sequence):
            dtypes = [dtypes]
        self.dtypes = [np.dtype(d) for d in dtypes]
        if not function_is_generator:
            self.func_cpu = getattr(module_cpu, function_name)
        else:
            # benchmark by generating all values
            def gen_cpu(*args, **kwargs):
                generator = getattr(module_cpu, function_name)(*args, **kwargs)
                return list(generator)

            self.func_cpu = gen_cpu
        self.kw_apply_parallel = apply_parallel_kwargs

        self.output_wrapper = output_wrapper

        self.module_name_cpu = module_cpu.__name__

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
        assert imaged.dtype == dtype
        self.args_cpu = (image,)

    def _index(self, name, var_kwargs, dtype=None, shape=None):
        index = name
        if var_kwargs:
            index += " ("
        params = []
        for k, v in var_kwargs.items():
            if isinstance(v, types.FunctionType):
                params.append(f"{k}={v.__name__}")
            elif isinstance(v, (np.ndarray)):
                params.append(f"{k}=array,shape={v.shape},dtype={v.dtype.name}")
            else:
                params.append(f"{k}={v}")
        if dtype is not None:
            params.append(f", {np.dtype(dtype).name}")
        if shape is not None:
            params.append(f"s={shape}")
        index += ", ".join(params)
        index.replace(",,", ",")
        if var_kwargs:
            index += ") "
        if self.index_str is not None:
            index += ", " + self.index_str
        return index

    def get_reps(self, func, args, kwargs, target_duration=2, cpu=True):
        if cpu:
            return dict(n_warmup=0, max_duration=target_duration)

        if not cpu:
            # dry run
            func(*args, **kwargs)
        # time 1 repetition
        tstart = time.time()
        out = func(*args, **kwargs)
        if hasattr(out, 'compute'):
            out.compute()
        dur = time.time() - tstart
        n_repeat = max(1, math.ceil(target_duration / dur))
        if cpu:
            n_warmup = 0
        else:
            n_warmup = max(1, math.ceil(n_repeat / 5))
        reps = dict(n_warmup=n_warmup, n_repeat=n_repeat)
        return reps

    def run_benchmark(self, duration=2, verbose=True):
        df = pd.DataFrame()
        self.df = df
        kw_lists = self.var_kwargs
        pdict = list(product_dict(**kw_lists))
        for dtype in self.dtypes:
            self.set_args(dtype)
            for i, var_kwargs1 in enumerate(pdict):
                # arr_index = indices[i]
                index = self._index(self.function_name, var_kwargs1)

                # transfer any arrays in kwargs to the appropriate device
                var_kwargs_cpu = var_kwargs1

                kw_cpu = {**self.fixed_kwargs_cpu, **var_kwargs_cpu}
                rep_kwargs_cpu = self.get_reps(
                    self.func_cpu, self.args_cpu, kw_cpu, duration, cpu=True
                )
                perf = repeat(self.func_cpu, self.args_cpu, kw_cpu, **rep_kwargs_cpu)

                if self.output_wrapper is None:
                    func_parallel = self.func_cpu
                else:
                    if self.output_wrapper == 'first_output_only':
                        def dummy(*args, **kwargs):
                            return self.func_cpu(*args, **kwargs)[0]
                        func_parallel = dummy

                args_apply_parallel = (func_parallel,) + (self.args_cpu[0],)   # assumes first argument is the image
                kw_apply_parallel = copy.deepcopy(self.kw_apply_parallel)
                kw_apply_parallel['extra_arguments'] = self.args_cpu[1:] if len(self.args_cpu) > 1 else {}
                kw_apply_parallel['extra_keywords'] = kw_cpu
                perf_apply_paralell = repeat(apply_parallel, args_apply_parallel, kw_apply_parallel, **rep_kwargs_cpu)

                df.at[index, "Dask (apply_parallel) accel"] = perf.cpu_times.mean() / perf_apply_paralell.cpu_times.mean()
                df.at[index, "shape"] = f"{self.shape}"
                # df.at[index,  "description"] = index
                df.at[index, "function_name"] = self.function_name
                df.at[index, "dtype"] = np.dtype(dtype).name
                df.at[index, "ndim"] = len(self.shape)
                df.at[index, "chunks"] = f"{self.chunks}"

                df.at[index, "CPU: host (mean)"] = perf.cpu_times.mean()
                df.at[index, "CPU: host (std)"] = perf.cpu_times.std()

                df.at[index, "Dask (CPU): host (mean)"] = perf_apply_paralell.cpu_times.mean()
                df.at[index, "Dask (CPU): host (std)"] = perf_apply_paralell.cpu_times.std()

                cmd = "cat /proc/cpuinfo"
                cpuinfo = subprocess.check_output(cmd, shell=True).strip()
                cpu_name = re.search("\nmodel name.*\n", cpuinfo.decode()).group(0).strip('\n')
                cpu_name = cpu_name.replace('model name\t: ', '')
                df.at[index, "CPU: DEV Name"] = [cpu_name for i in range(len(df))]

                # accelerations[arr_index] = df.at[index,  "GPU accel"]
                if verbose:
                    print(df.loc[index])

        results = {}
        results["full"] = df
        results["var_kwargs_names"] = list(self.var_kwargs.keys())
        results["var_kwargs_values"] = list(self.var_kwargs.values())
        results["function_name"] = self.function_name
        results["module_name_cpu"] = self.module_name_cpu
        return results
