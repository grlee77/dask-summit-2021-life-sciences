import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

font_size = 18
rcParams['font.size'] = font_size
rcParams['axes.titleweight'] = 'bold'


labels = [
    'median filter: size (3, 3, 3)',
    'median filter: size (5, 5, 5)',
    'median filter: size (7, 7, 7)',
    # 'map coordinates\n(3D, order=1)',
    ]

ben_cpu_means = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.8601,   # median 3x3x3
    3.9334,   # median 5x5x5
    11.387,    # median 7x7x7
]

greg_cpu_means = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    2.5514,   # median 3x3x3
    9.7912,   # median 5x5x5
    26.225,   # median 7x7x7
]

greg_dask_cpu_means = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.3359,   # median 3x3x3
    1.0804,   # median 5x5x5
    2.7793,   # median 7x7x7
]

gpu_means_1080ti = [
    0.01033,   # median 3x3x3
    0.4321,   # median 5x5x5
    2.3030,   # median 7x7x7
]


gpu_means_V100 =  [  # V100 results
]

gpu_means_A100 =  [  # A100 results
    0.0009652,   # median 3x3x3
    0.08003,   # median 5x5x5
    0.114466,    # median 7x7x7
]


x = np.arange(len(labels))  # the label locations

n_gpus = 0  # number of GPU models to include in the plot. Must be between 1 and 3
log_scale = n_gpus > 0

ones = np.ones_like(gpu_means_A100)
accels_dask = np.array(greg_cpu_means) / np.array(greg_dask_cpu_means)
accels_ti = np.array(greg_cpu_means) / np.array(gpu_means_1080ti)
accels_a100 = np.array(greg_cpu_means) / np.array(gpu_means_A100)
[print(f"{name}: acceleration = {acc}") for name, acc in zip(labels, accels_a100)]


if n_gpus == 0:
    # Plot A100 results vs. CPU only
    width = 0.43  # the width of the bars
    figsize = [18, 5.58]
    fig, ax = plt.subplots(figsize=figsize)

    rects1 = ax.bar(x - width / 2, ones, width, label='SciPy (CPU): Intel Core i9-7900X', color='#FFB043')
    rects2 = ax.bar(x + width / 2, accels_dask, width, label='dask-image (CPU): Intel Core i9-7900X', color='#6A16F8')

elif n_gpus == 1:
    # A100 and V100 results vs. CPU
    figsize = [22, 6.54]
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.29  # the width of the bars
    rects1 = ax.bar(x - width, ones, width, label='SciPy (CPU): Intel Core i9-7900X', color='#FFB043')
    rects2 = ax.bar(x, accels_dask, width, label='dask-image (CPU): Intel Core i9-7900X', color='#2CC9D9')
    rects3 = ax.bar(x + width, accels_ti, width, label='CuPy (GPU): NVIDIA GTX 1080 Ti', color='#6A16F8')

elif n_gpus == 2:
    # A100, V100 and GTX-1080 Ti results vs. CPU
    figsize = [30, 7.36]
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.23  # the width of the bars
    rects1 = ax.bar(x - 1.5 * width, ones, width, label='SciPy (CPU): Intel Core i9-7900X', color='#FFB043')
    rects2 = ax.bar(x - 0.5 * width, accels_dask, width, label='dask-image (CPU): Intel Core i9-7900X', color='#D028C8')
    rects3 = ax.bar(x + 0.5 * width, accels_ti, width, label='CuPy (GPU): NVIDIA GTX 1080 Ti', color='#2CC9D9')
    rects4 = ax.bar(x + 1.5 * width, accels_a100, width, label='CuPy (GPU): NVIDIA A100', color='#6A16F8')
else:
    raise NotImplementedError("only 3 GPU models have results stored in this script")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance Gain', fontdict=dict(fontweight='bold', fontsize=font_size))
if n_gpus > 0:
    ax.set_title('3D Median Filtering Performance: SciPy / dask-image / CuPy')
else:
    ax.set_title('3D Median Filtering Performance: dask-image vs. SciPy')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontdict=dict(fontweight='bold', fontsize=font_size))
ax.set_ylim([0, 5.0])
ax.legend()

if log_scale:
    ax.set_yscale('log')
    #ax.set_ylim([0, 900.0])
    ax.set_ylim([0, 5000.0])
    ax.set_yticks([0.1, 1.0, 10.0, 100.0, 1000.0])
    ax.set_yticklabels(['0.1x', '1x', '10x', '100x', '1000x'])
    max_label_y = np.inf
else:
    max_label_y = 10
    ax.set_ylim([0, 15.0])


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 1e3:
            v  = '{:0.4g}x'.format(height)
        else:
            v  = '{:0.3g}x'.format(height)
        ax.annotate(v,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
if n_gpus >= 0:
    autolabel(rects2)
if n_gpus >= 1:
    autolabel(rects3)
if n_gpus >= 2:
    autolabel(rects4)

fig.tight_layout()

plt.show()
