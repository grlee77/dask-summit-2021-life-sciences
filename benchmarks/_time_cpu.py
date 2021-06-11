# adapted from CuPy. see ../LICENSE-3rdparty.txt

import math as _math
import time as _time

import numpy as _numpy


class _PerfCaseResult:
    """ An obscure object encompassing timing results recorded by
    :func:`~cupyx.time.repeat`. Simple statistics can be obtained by converting
    an instance of this class to a string.

    .. warning::
        This API is currently experimental and subject to change in future
        releases.

    """

    def __init__(self, name, ts):
        assert ts.ndim == 1
        self.name = name
        self._ts = ts

    @property
    def cpu_times(self):
        """ Returns an array of CPU times of size ``n_repeat``. """
        return self._ts

    @staticmethod
    def _to_str_per_item(device_name, t):
        assert t.ndim == 1
        assert t.size > 0
        t_us = t * 1e6

        s = '    {}:{:9.03f} us'.format(device_name, t_us.mean())
        if t.size > 1:
            s += '   +/-{:6.03f} (min:{:9.03f} / max:{:9.03f}) us'.format(
                t_us.std(), t_us.min(), t_us.max())
        return s

    def to_str(self):
        results = [self._to_str_per_item('CPU', self._ts)]
        return '{:<20s}:{}'.format(self.name, ' '.join(results))

    def __str__(self):
        return self.to_str()


def repeat(
        func, args=(), kwargs={}, n_repeat=10000, *,
        name=None, n_warmup=10, max_duration=_math.inf):
    """ Timing utility for measuring time spent by CPU.

    This function is a very convenient helper for setting up a timing test.

    Args:
        func (callable): a callable object to be timed.
        args (tuple): positional argumens to be passed to the callable.
        kwargs (dict): keyword arguments to be passed to the callable.
        n_repeat (int): number of times the callable is called. Increasing
            this value would improve the collected statistics at the cost
            of longer test time.
        name (str): the function name to be reported. If not given, the
            callable's ``__name__`` attribute is used.
        n_warmup (int): number of times the callable is called. The warm-up
            runs are not timed.
        max_duration (float): the maximum time (in seconds) that the entire
            test can use. If the taken time is longer than this limit, the test
            is stopped and the statistics collected up to the breakpoint is
            reported.

    Returns:
        :class:`_PerfCaseResult`: an object collecting all test results.

    .. warning::
        This API is currently experimental and subject to change in future
        releases.

    """

    if name is None:
        name = func.__name__

    if not callable(func):
        raise ValueError('`func` should be a callable object.')
    if not isinstance(args, tuple):
        raise ValueError('`args` should be of tuple type.')
    if not isinstance(kwargs, dict):
        raise ValueError('`kwargs` should be of dict type.')
    if not isinstance(n_repeat, int):
        raise ValueError('`n_repeat` should be an integer.')
    if not isinstance(name, str):
        raise ValueError('`name` should be a string.')
    if not isinstance(n_warmup, int):
        raise ValueError('`n_warmup` should be an integer.')
    if not _numpy.isreal(max_duration):
        raise ValueError('`max_duration` should be given in seconds')

    return _repeat(
        func, args, kwargs, n_repeat, name, n_warmup, max_duration)


def _repeat(
        func, args, kwargs, n_repeat, name, n_warmup, max_duration):

    for i in range(n_warmup):
        func(*args, **kwargs)

    cpu_times = []
    duration = 0
    for i in range(n_repeat):
        t1 = _time.perf_counter()

        func(*args, **kwargs)

        t2 = _time.perf_counter()
        cpu_time = t2 - t1
        cpu_times.append(cpu_time)

        duration += cpu_time
        if duration > max_duration:
            break

    ts = _numpy.asarray(cpu_times, dtype=_numpy.float64)
    return _PerfCaseResult(name, ts)
